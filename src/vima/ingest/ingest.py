import numpy as numpy
import numpy as np
import xarray as xr
import cv2 as cv2
import scanpy as sc
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
import gc, os
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)
from . import util, dimreduce

def visualize_pixels(pixels, ntoplot, input, colorby, include_pca_plot=False):
    pcs = [c for c in pixels.columns if c.startswith('PC')]
    metavars = [c for c in pixels.columns if c not in pcs]
    np.random.seed(0)
    ix = np.random.choice(len(pixels), replace=False, size=ntoplot)
    toplot = pixels.iloc[ix]
    obs = toplot[metavars].copy()
    obs.index = obs.index.astype(str)
    toplot_ad = ad.AnnData(X=toplot[pcs], obs=obs)
    sc.pp.neighbors(toplot_ad, use_rep='X')
    sc.tl.umap(toplot_ad)
    
    for metavar in colorby:
        if include_pca_plot:
            sns.scatterplot(x='PC1', y='PC2', hue=metavar, data=toplot, palette='Set1', s=1, legend=False)
            plt.title(metavar)
            plt.show()
        sc.pl.umap(toplot_ad, color=metavar, legend_loc=None, frameon=False,
                   title=f'pixels UMAPed using {input}, colored by {metavar}')

    return toplot_ad

def add_covs(pca, sid_to_covs):
    if sid_to_covs is not None:
        cov_names = list(sid_to_covs.columns)
    else:
        cov_names = []
    for cov_name in cov_names:
        pca[cov_name] = pca['sid'].map(sid_to_covs[cov_name])
    return ['sid'] + cov_names

def pca_pixels(outdir, repname, nmetamarkers=10, plot=True, npixels_to_plot=50000, sid_to_covs=None):
    # prepare directory structure
    masksdir = f'{outdir}/masks'
    normeddir = f'{outdir}/normalized'
    processeddir = f'{outdir}/{repname}'
    os.makedirs(processeddir, exist_ok=True)

    # prepare
    sids = [os.path.splitext(f)[0]
        for f in os.listdir(normeddir) if f.endswith('.nc') and not f.startswith('.')]

    # create metapixels for more accurate PCA
    metapixels, npixels = dimreduce.metapixels_allsamples(normeddir, masksdir, sids, plot=plot)

    # PCA the metapixels
    loadings, C, allmp = dimreduce.pca_metapixels(metapixels.values(), nmetamarkers, plot=plot)
    loadings.to_feather(f'{processeddir}/_pcloadings.feather')
    del metapixels, allmp; gc.collect()

    # apply the PC loadings to plain pixels
    pca = dimreduce.pca_pixels(normeddir, masksdir, loadings, sids)

    # add covariates
    cov_names = add_covs(pca, sid_to_covs)

    if plot:
        visualize_pixels(pca, npixels_to_plot, 'metamarkers', cov_names)
    return pca

def harmonize(allpixels_pca, sid_to_covs=None, npixels_to_plot=50000, plot=True):
    import harmonypy as hm

    harmony_cov_names = add_covs(allpixels_pca, sid_to_covs)
    pcs = [c for c in allpixels_pca.columns if c.startswith('PC')]

    print('Running Harmony...')
    harmony_out = hm.run_harmony(allpixels_pca[pcs].values, allpixels_pca, harmony_cov_names)

    harmpixels = allpixels_pca.copy()
    harmpixels[pcs] = harmony_out.Z_corr

    if plot:
        visualize_pixels(harmpixels, npixels_to_plot, 'harm. metamarkers', harmony_cov_names)

    return harmpixels

def write_harmonized(outdir, repname, harmpixels):
    masksdir = f'{outdir}/masks'
    processeddir = f'{outdir}/{repname}'
    pcs = [c for c in harmpixels.columns if c.startswith('PC')]
    hpcs = ['h'+c for c in pcs]
    for sid in pb(harmpixels.sid.unique()):
        mask = xr.open_dataarray(f'{masksdir}/{sid}.nc')
        pl = harmpixels[harmpixels.sid == sid]
        s_ = np.zeros((*mask.shape, len(hpcs)))
        s_[mask.data] = pl[pcs].values
        s = xr.DataArray(s_,
             dims=['y', 'x', 'marker'],
             coords={'x': mask.x, 'y': mask.y, 'marker': hpcs})
        s.name = sid
        s.to_netcdf(f'{processeddir}/{sid}.nc', encoding={s.name: util.compression}, engine="netcdf4")
        gc.collect()

def sanity_checks(outdir, repname, sid_to_covs=None):
    processeddir = f'{outdir}/{repname}'
    sids = [os.path.splitext(f)[0]
        for f in os.listdir(processeddir) if f.endswith('.nc')]

    print('all PCs of one sample')
    s = xr.open_dataarray(f'{processeddir}/{sids[0]}.nc').astype(np.float32)
    s.plot(col='marker', col_wrap=5, vmin=-10, vmax=10, cmap='seismic')
    plt.show()

    print('histogram of each pc')
    ss = [xr.open_dataarray(f'{processeddir}/{sid}.nc').astype(np.float32) for sid in sids]
    nmms = len(ss[0].marker)
    harmpixels = np.concatenate([s.data.reshape((-1, nmms)) for s in ss])
    harmpixels = harmpixels[(harmpixels != 0).sum(axis=1) > 0]
    plt.figure(figsize=(3*4, 2*int(np.ceil(nmms/4))))
    for i in pb(range(nmms)):
        plt.subplot(int(np.ceil(nmms/4)), 4, i+1)
        plt.hist(harmpixels[:,i], bins=1000)
    plt.tight_layout()
    plt.show()
    del ss
    del harmpixels
    gc.collect()

    print('PC1 of several samples')
    fig, axs = plt.subplots(len(sids[::5])//5 + 1, 5, figsize=(16, 4*(len(sids[::5])//5 + 1)))
    for sid, ax in zip(sids[::3], axs.flatten()):
        s = xr.open_dataarray(f'{processeddir}/{sid}.nc').astype(np.float32)
        vmax = np.percentile(np.abs(s.sel(marker='hPC1').data), 99)
        s.sel(marker='hPC1').plot(ax=ax, cmap='seismic', vmin=-vmax, vmax=vmax, add_colorbar=False)
        ax.set_title(sid)
        gc.collect()
    plt.tight_layout()
    plt.show()