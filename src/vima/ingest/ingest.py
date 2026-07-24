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

def _lisi_ratio(conn, labels):
    from scipy.sparse import issparse, diags as sp_diags
    labels = np.asarray(labels)
    unique, inv = np.unique(labels, return_inverse=True)
    n_cats = len(unique)
    if n_cats < 2:
        return float('nan')
    n = conn.shape[0]
    L = np.zeros((n, n_cats))
    L[np.arange(n), inv] = 1.0
    rs = np.asarray(conn.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    freq = (sp_diags(1.0 / rs) @ conn) @ L
    if issparse(freq):
        freq = freq.toarray()
    mean_lisi = (1.0 / (freq ** 2).sum(axis=1)).mean()
    global_q = np.bincount(inv).astype(float) / n
    return mean_lisi, 1.0 / (global_q ** 2).sum()

def visualize_pixels(pixels, ntoplot, input, colorby, include_pca_plot=False):
    """
    UMAP a random subset of pixels, colored by covariates, and report mixing.

    Embeds ``ntoplot`` randomly sampled pixels using their PC columns and plots
    the UMAP colored by each covariate in ``colorby``. For each covariate,
    prints an average LISI ratio quantifying how well its categories mix (the
    printed baseline marks perfect mixing, 1 marks no mixing).

    Parameters
    ----------
    input
        Label for the representation being embedded, used in plot titles.
    colorby
        Covariate column names to color the UMAP by.

    Returns
    -------
    AnnData
        The embedded subset, with the UMAP and neighbor graph attached.
    """
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
        
        # print LISI ratio
        n_unique = toplot_ad.obs[metavar].nunique()
        if n_unique < 2 or n_unique > 1000:
            print(f'Avg LISI ({metavar}): skipped ({n_unique} unique values)')
        else:
            lisi, baseline = _lisi_ratio(toplot_ad.obsp['connectivities'], toplot_ad.obs[metavar])
            print(f'\033[91mAvg LISI ({metavar}): {lisi:.2f}  [{baseline:.2f} = perfect mixing, 1 = not mixed]\033[0m')
            print()

    return toplot_ad

def add_covs(pca, sid_to_covs):
    """
    Attach per-sample covariate columns to a pixel table (in place).

    Maps each covariate in ``sid_to_covs`` onto the pixel table via its 'sid'
    column.

    Parameters
    ----------
    sid_to_covs
        DataFrame indexed by sample ID with one column per covariate, or None.

    Returns
    -------
    list
        Covariate names including 'sid'.
    """
    if sid_to_covs is not None:
        cov_names = list(sid_to_covs.columns)
    else:
        cov_names = []
    for cov_name in cov_names:
        pca[cov_name] = pca['sid'].map(sid_to_covs[cov_name])
    return ['sid'] + cov_names

def pca_pixels(outdir, repname, nmetamarkers=10, plot=True, npixels_to_plot=50000,
               total_n_metapixels=2_000_000, sid_to_covs=None):
    """
    Reduce normalized pixels to a small set of PCA "meta-markers".

    Fits PCA on spatially pooled metapixels across all samples, then projects
    every pixel onto the top `nmetamarkers` components. This compresses the gene
    panel into a compact representation for model training.

    Parameters
    ----------
    outdir
        Directory written by `prepare_merfish`/`prepare_xenium5k`.
    repname
        Name of the representation; PC loadings are saved under
        ``outdir/repname``.
    nmetamarkers
        Number of principal components (meta-markers) to keep.
    total_n_metapixels
        Target number of pooled metapixels used to fit the PCA.
    sid_to_covs
        Optional per-sample covariates, used only for the diagnostic plots.

    Returns
    -------
    tuple
        ``(allpixels_pca, pc_loadings)``: the projected pixels (with a 'sid'
        column) and the gene-by-component loading matrix.
    """
    # prepare directory structure
    masksdir = f'{outdir}/masks'
    normeddir = f'{outdir}/normalized'
    processeddir = f'{outdir}/{repname}'
    os.makedirs(processeddir, exist_ok=True)

    # prepare
    sids = [os.path.splitext(f)[0]
        for f in os.listdir(normeddir) if f.endswith('.nc') and not f.startswith('.')]

    # create metapixels for more accurate PCA
    metapixels, npixels = dimreduce.metapixels_allsamples(normeddir, masksdir, sids,
                                                          total_n_metapixels=total_n_metapixels,
                                                          plot=plot)

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
    return pca, loadings

def harmonize(allpixels_pca, sid_to_covs=None, npixels_to_plot=50000, plot=True):
    """
    Batch-correct meta-marker pixels across samples with Harmony.

    Parameters
    ----------
    allpixels_pca
        Projected pixels from `pca_pixels`, with a 'sid' column.
    sid_to_covs
        Optional per-sample covariates to integrate over in addition to sample
        ID.

    Returns
    -------
    DataFrame
        Copy of `allpixels_pca` with the PC columns replaced by their
        Harmony-corrected values.
    """
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
    """
    Write harmonized meta-marker pixels to per-sample ``.nc`` files.

    Rasterizes the corrected pixels from `harmonize` back into ``(y, x, marker)``
    matrices under ``outdir/repname``, ready to be loaded with `read_samples`.
    """
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
        mask.close(); del mask
        gc.collect()

def sanity_checks(outdir, repname, npcs=1, nskip=3):
    """
    Plot diagnostics of the harmonized meta-marker representation.

    Shows all meta-markers of one sample, a histogram of each meta-marker
    pooled over all samples, and spatial maps of the first ``npcs`` meta-markers
    across a subset of samples.

    Parameters
    ----------
    repname
        Representation subdirectory under ``outdir`` written by
        `write_harmonized`.
    npcs
        Number of leading meta-markers to map spatially across samples.
    nskip
        Plot every ``nskip``-th sample in the spatial maps.
    """
    processeddir = f'{outdir}/{repname}'
    sids = [os.path.splitext(f)[0]
        for f in os.listdir(processeddir) if f.endswith('.nc')]

    print('all PCs of one sample')
    da = xr.open_dataarray(f'{processeddir}/{sids[0]}.nc')
    s = da.astype(np.float32)
    da.close(); del da
    s.plot(col='marker', col_wrap=5, vmin=-10, vmax=10, cmap='seismic')
    plt.show()
    del s

    print('histogram of each pc')
    da0 = xr.open_dataarray(f'{processeddir}/{sids[0]}.nc')
    nmms = len(da0.marker)
    da0.close(); del da0
    chunks = []
    for sid in sids:
        da = xr.open_dataarray(f'{processeddir}/{sid}.nc')
        chunks.append(da.astype(np.float32).data.reshape((-1, nmms)))
        da.close(); del da
    harmpixels = np.concatenate(chunks)
    del chunks
    harmpixels = harmpixels[(harmpixels != 0).sum(axis=1) > 0]
    plt.figure(figsize=(3*4, 2*int(np.ceil(nmms/4))))
    for i in pb(range(nmms)):
        plt.subplot(int(np.ceil(nmms/4)), 4, i+1)
        plt.hist(harmpixels[:,i], bins=1000)
    plt.tight_layout()
    plt.show()
    del harmpixels
    gc.collect()

    for i in range(1, npcs+1):
        print(f'PC{i} of several samples')
        fig, axs = plt.subplots(len(sids[::nskip])//5 + 1, 5, figsize=(16, 4*(len(sids[::nskip])//5 + 1)))
        flat_axs = axs.flatten()
        for ax in flat_axs:
            ax.set_visible(False)
        for sid, ax in zip(sids[::nskip], flat_axs):
            ax.set_visible(True)
            da = xr.open_dataarray(f'{processeddir}/{sid}.nc')
            s = da.astype(np.float32)
            da.close(); del da
            vmax = np.percentile(np.abs(s.sel(marker=f'hPC{i}').data), 99)
            s.sel(marker=f'hPC{i}').plot(ax=ax, cmap='seismic', vmin=-vmax, vmax=vmax, add_colorbar=False)
            ax.set_title(sid)
            del s; gc.collect()
        plt.tight_layout()
        plt.show()