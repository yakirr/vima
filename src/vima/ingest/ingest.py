import numpy as numpy
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import xarray as xr
import cv2 as cv2
from skimage.filters import threshold_otsu
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gc, os, subprocess
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)

compression = {'zlib': True, 'complevel': 2} # settings for writing xarrays

###########################################
# utility functions
###########################################
def xr_to_pixellist(s, mask):
    return s.data[mask.data]

def set_pixels(s, mask, pl):
    s.data[mask.data] = pl

def ar():
    plt.gca().set_aspect('equal')

###########################################
# for creating raw pixel files
###########################################
def transcriptlist_to_pixellist(transcriptlist, x_colname='global_x', y_colname='global_y', gene_colname='gene', pixel_size=10):
    # adds dummy rows such that there is at least one entry for every possible x- and y- value
    # between the min and max values
    def complete(pl, colname, genes, fill=0., verbose=True):
        vals = np.sort(pl[colname].unique())
        min_col = vals.min() // 1
        max_col = vals.max() // 1
        delta = int(min(vals[1:] - vals[:-1]))
        full_range = list(np.arange(min_col, max_col + 1, delta))
        locs_toadd = np.setdiff1d(full_range, vals)
        if verbose: print(f'\tadding {colname}={locs_toadd}')
        toadd = pl.iloc[:len(locs_toadd)].copy()
        toadd[colname] = locs_toadd
        toadd[genes] = fill
        return pd.concat([pl, toadd], axis=0, ignore_index=True)

    transcriptlist = transcriptlist[[x_colname, y_colname, gene_colname]].copy()
    transcriptlist['pixel_x'] = (transcriptlist[x_colname] / pixel_size).astype(int) * pixel_size
    transcriptlist['pixel_y'] = (transcriptlist[y_colname] / pixel_size).astype(int) * pixel_size

    pixels = transcriptlist.groupby(['pixel_x', 'pixel_y'])[gene_colname].value_counts().unstack(fill_value=0)
    pixels.reset_index(inplace=True)
    pl = pixels.rename_axis(None, axis=1)
    genes = pl.columns[2:]

    return complete(complete(pl, 'pixel_x', genes), 'pixel_y', genes)

def pixellist_to_pixelmatrix(pl, markers):
    # pivot in pandas
    s = pd.pivot_table(pl, values=markers, index='pixel_y', columns='pixel_x').fillna(0)
    s.columns.names = ['markers', 'pixel_x']

    # convert to xarray
    s = df_to_xarray32(s)
    print('sample shape:', s.shape)

    return s

def df_to_xarray32(df):
    markers = df.columns.get_level_values('markers').unique()
    return xr.DataArray(
            df.values.reshape((len(df), len(markers), -1)).transpose(0,2,1),
            coords={'x': df.columns.get_level_values('pixel_x').unique().values, 'y': df.index.values, 'marker': markers.values},
            dims=['y', 'x', 'marker']
        ).astype(np.float32)

def downsample(sample, factor, aggregate=np.mean):
    pad_width = (
        (int(factor - sample.shape[0] % factor), 0),
        (int(factor - sample.shape[1] % factor), 0),
        (0,0))
    sample = np.pad(sample, pad_width, mode='constant', constant_values=0)
    smaller = sample.reshape(sample.shape[0], sample.shape[1]//factor, factor, sample.shape[2])
    smaller = aggregate(smaller, axis=2)
    smaller = smaller.reshape(smaller.shape[0]//factor, factor, smaller.shape[1], smaller.shape[2])
    smaller = aggregate(smaller, axis=1)
    return smaller

def hiresarray_to_downsampledxarray(sample, name, factor, pixelsize, markers):
    sample = downsample(sample, factor)
    sample = xr.DataArray(
            sample,
            coords={'x': np.arange(sample.shape[1])*factor*pixelsize, 'y': np.arange(sample.shape[0])*factor*pixelsize, 'marker': markers},
            dims=['y', 'x', 'marker']
        ).astype(np.float32)
    sample.name = name
    return sample

###########################################
# processing raw pixel files
###########################################
def foreground_mask_st(s, min_ntranscripts=10):
    totals = s.sum(dim='marker')
    mask = totals > min_ntranscripts
    return mask

def foreground_mask_ihc(s, real_markers, neg_ctrls, not_imaged_thresh, artifact_thresh, transform=lambda x:x, thresholding_method=threshold_otsu,
        neg_ctrl_pseudocount=0, blur_width=5):
    totals = (s.sel(marker=real_markers).sum(dim='marker') / (s.sel(marker=neg_ctrls).sum(dim='marker') + len(neg_ctrls) + neg_ctrl_pseudocount))
    totals = transform(cv2.GaussianBlur(totals.data, (blur_width, blur_width),0))
    valid_pixels = totals[(totals > not_imaged_thresh) & (totals < artifact_thresh)]
    t = thresholding_method(valid_pixels)
    
    return xr.DataArray(((totals > t) & (totals < artifact_thresh)).astype('bool'),
                coords={'x': s.x, 'y': s.y},
                dims=['y','x'], name=s.name)

def foreground_mask_codex(s, real_markers, neg_ctrls, blur_width=5):
    # compute totals
    totals = s.sel(marker=real_markers).sum(dim='marker')
    totals = np.log1p(totals)
    totals -= totals.min()
    totals /= (totals.max()/255)
    totals = totals.astype('uint16')

    # determine foreground vs background
    blurred = cv2.GaussianBlur(totals.data,(blur_width, blur_width),0)
    _, mask = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return xr.DataArray(mask.astype('bool'),
                coords={'x': totals.x, 'y': totals.y},
                dims=['y','x'], name=s.name)

def write_masks(pixelsdir, outdir, get_foreground, sids, plot=True, vmax=30):
    for sid in sids:
        print(sid, end=': ')
        s = xr.load_dataarray(f'{pixelsdir}/{sid}.nc').astype(np.float32)
        
        # make mask and save
        mask = get_foreground(s)
        print(f'{mask.values.sum()} ({100*mask.values.sum()/(mask.shape[0]*mask.shape[1]):.0f}%) pixels non-empty', end='| ')
        mask.to_netcdf(f'{outdir}/{sid}.nc', encoding={mask.name: compression}, engine="netcdf4")

        if plot:
            s.sum(dim='marker').plot(cmap='Reds', vmin=0, vmax=vmax); ar()
            mask.plot(alpha=0.5, vmin=0, vmax=1, cmap='gray', add_colorbar=False)
            plt.show()
            
            subset = s.where(mask, other=0).sel(marker=s.marker[::10])
            norm = mcolors.Normalize(vmin=subset.data.min(), vmax=0.95*subset.data.max())
            subset.plot(col='marker', col_wrap=4, norm=norm)
            plt.show()
        
        gc.collect()

def get_sumstats_nonst(norm, pixels):
    pixels = norm(pixels)[1]
    ntranscripts = pixels.sum(axis=1, dtype=np.float64)
    med_ntranscripts = np.median(ntranscripts)
    pixels = np.log1p(med_ntranscripts * pixels / (ntranscripts[:,None] + 1e-6)) # adding to denominator in case pixel is all 0s
    means = pixels.mean(axis=0, dtype=np.float64)
    stds = pixels.std(axis=0, dtype=np.float64)
    return {'means':means, 'stds':stds, 'med_ntranscripts':med_ntranscripts}

def normalize_nonst(norm, mask, s, med_ntranscripts=None, means=None, stds=None):
    s = s.where(mask, other=0)
    pl = xr_to_pixellist(s, mask)
    markers_to_keep, pl = norm(pl)
    pl = np.log1p(med_ntranscripts * pl / (pl.sum(axis=1)[:,None] + 1e-6)) # adding to denominator in case pixel is all 0s
    pl -= means
    pl /= stds
    s = s.sel(marker=markers_to_keep)
    set_pixels(s, mask, pl)
    s.attrs['med_ntranscripts'] = med_ntranscripts
    s.attrs['means'] = means
    s.attrs['stds'] = stds
    return s

def get_sumstats_st(pixels):
    ntranscripts = pixels.sum(axis=1, dtype=np.float64)
    med_ntranscripts = np.median(ntranscripts)
    pixels = np.log1p(med_ntranscripts * pixels / ntranscripts[:,None])
    means = pixels.mean(axis=0, dtype=np.float64)
    stds = pixels.std(axis=0, dtype=np.float64)
    return {'means':means, 'stds':stds, 'med_ntranscripts':med_ntranscripts}

def normalize_st(mask, s, med_ntranscripts=None, means=None, stds=None):
    s = s.where(mask, other=0)
    pl = xr_to_pixellist(s, mask)
    pl = np.log1p(med_ntranscripts * pl / pl.sum(axis=1)[:,None])
    pl -= means
    pl /= stds
    set_pixels(s, mask, pl)
    s.attrs['med_ntranscripts'] = med_ntranscripts
    s.attrs['means'] = means
    s.attrs['stds'] = stds
    return s

def normalize_allsamples(pixelsdir, masksdir, outdir, sids, get_sumstats=get_sumstats_st, normalize=normalize_st):
    print('\nreading all non-empty pixels')
    pixels = np.concatenate([
        xr_to_pixellist(
            xr.open_dataarray(f'{pixelsdir}/{sid}.nc').astype(np.float32),
            xr.open_dataarray(f'{masksdir}/{sid}.nc')
            )
        for sid in pb(sids)])
    gc.collect()

    print('computing sumstats')
    sumstats = get_sumstats(pixels)
    del pixels; gc.collect()
    
    print('normalizing and writing')
    for sid in pb(sids):
        s = normalize(
            xr.open_dataarray(f'{masksdir}/{sid}.nc'),
            xr.open_dataarray(f'{pixelsdir}/{sid}.nc').astype(np.float32),
            **sumstats)
        s.to_netcdf(f'{outdir}/{sid}.nc', encoding={s.name: compression}, engine="netcdf4")

###########################################
# dimensionality reduction and integration
###########################################
def metapixels_allsamples(normedpixelsdir, masksdir, sids, plot=True, ncols=8):
    def cdf(v, ax):
        sorted_data = np.sort(v)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf)

    all_metapixels = {}
    all_npixels = {}

    if plot:
        nrows = int(np.ceil(len(sids)/ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(2*ncols,1.5*nrows))
        axs = axs.reshape((nrows, -1))

    print('creating metapixels prior to PCA')
    for i, sid in pb(enumerate(sids)):
        all_metapixels[sid], all_npixels[sid] = metapixels(
            xr.open_dataarray(f'{normedpixelsdir}/{sid}.nc').astype(np.float32),
            xr.open_dataarray(f'{masksdir}/{sid}.nc'))

        # visualize distribution of num non-empty pixels per metapixel in this sample
        if plot:
            ax = axs[i // ncols, i % ncols]
            cdf(all_npixels[sid], ax)
            ax.set_title(sid)
        gc.collect()

    if plot:
        plt.tight_layout()
        plt.show()

    return all_metapixels, all_npixels

def metapixels(s, mask, npixels_thresh=0):
    markers = s.marker.values

    # make metapixels and compute how many non-empty pixels and transcripts are in each metapixel
    kernel = np.ones((5,5),np.float32)
    mp = cv2.filter2D(s.data, -1, kernel)
    npixels = cv2.filter2D(mask.data.astype('float32'), -1, kernel)

    # filter out metapixels with few non-empty pixels
    metapixels_mask = npixels > npixels_thresh

    # divide each metapixel by the # of non-empty pixels that contributed to it and return
    return pd.DataFrame(data=mp[metapixels_mask] / npixels[metapixels_mask][:,None], columns=markers), npixels[metapixels_mask]

# mps should be an array of dataframes containing metapixels
def pca_metapixels(mps, k, plot=True):
    print('merging and standardizing metapixels')
    allmp = pd.concat(mps)
    allmp -= allmp.values.mean(axis=0, dtype=np.float64)
    allmp /= allmp.values.std(axis=0, dtype=np.float64)
    allmp = ad.AnnData(X=allmp)
    C = np.corrcoef(allmp.X[::max(1,(len(allmp)//50000))].T)
    print(allmp.shape)

    print('performing PCA')
    sc.tl.pca(allmp, n_comps=k)
    loadings = pd.DataFrame(data=allmp.varm['PCs'], columns=[f'PC{i}' for i in range(1,k+1)], index=allmp.var_names)

    if plot:
        plt.figure(figsize=(30,2))
        plt.imshow(loadings.T, cmap='seismic', vmin=-0.5, vmax=0.5)
        plt.xticks(range(len(loadings)), loadings.index, rotation=90)
        plt.show()

    return loadings, C, allmp

def pca_pixels(normedixelsdir, masksdir, pcloadings, sids, plot=True, npixels_to_plot=50000, colorby=['sid']):
    print('reading in pixels')
    pls = np.concatenate([
        xr_to_pixellist(
            xr.open_dataarray(f'{normedixelsdir}/{sid}.nc').astype(np.float32),
            xr.open_dataarray(f'{masksdir}/{sid}.nc'))
        for sid in pb(sids)])
    sid_labels = np.concatenate([
        np.array([sid] * xr.open_dataarray(f'{masksdir}/{sid}.nc').sum().item())
        for sid in sids
        ])
    print('applying dimensionality reduction')
    allpixels_pca = pd.DataFrame(
        pls.dot(pcloadings),
        columns=[f'PC{i}' for i in range(1,pcloadings.shape[1]+1)]
        )
    allpixels_pca['sid'] = sid_labels
    del pls; gc.collect()

    if plot:
        print('visualizing')
        visualize_pixels(allpixels_pca, npixels_to_plot, 'metamarkers', colorby)

    return allpixels_pca

def visualize_pixels(pixels, ntoplot, input, colorby, include_pca_plot=False):
    pcs = [c for c in pixels.columns if c.startswith('PC')]
    metavars = [c for c in pixels.columns if c not in pcs]
    np.random.seed(0)
    ix = np.random.choice(len(pixels), replace=False, size=ntoplot)
    toplot = pixels.iloc[ix]
    toplot_ad = ad.AnnData(
        X=toplot[pcs],
        obs=toplot[metavars])
    sc.pp.neighbors(toplot_ad, use_rep='X')
    sc.tl.umap(toplot_ad)
    
    for metavar in colorby:
        if include_pca_plot:
            sns.scatterplot(x='PC1', y='PC2', hue=metavar, data=toplot, palette='Set1', s=1, legend=False)
            plt.title(metavar)
            plt.show()
        sc.pl.umap(toplot_ad, color=metavar, legend_loc=None,
                   title=f'pixels UMAPed using {input}, colored by {metavar}')

    return toplot_ad

###########################################
# user-facing interface
###########################################
import glob
import tempfile

def preprocess(outdir, repname, get_foreground, get_sumstats, normalize,
                nmetamarkers=10, plot=False):
    # prepare directory structure
    countsdir = f'{outdir}/counts'
    normeddir = f'{outdir}/normalized'
    masksdir = f'{outdir}/masks'
    processeddir = f'{outdir}/{repname}'
    os.makedirs(normeddir, exist_ok=True)
    os.makedirs(masksdir, exist_ok=True)
    os.makedirs(processeddir, exist_ok=True)

    # prepare
    sids = [os.path.splitext(f)[0]
        for f in os.listdir(countsdir) if f.endswith('.nc')]

    # create and write masks
    write_masks(countsdir, masksdir, get_foreground, sids, plot=plot)

    # create normalized pixels
    normalize_allsamples(countsdir, masksdir, normeddir, sids,
                               get_sumstats=get_sumstats,
                               normalize=normalize)

    # create metapixels for more accurate PCA
    metapixels, npixels = metapixels_allsamples(normeddir, masksdir, sids, plot=plot)

    # PCA the metapixels
    loadings, C, allmp = pca_metapixels(metapixels.values(), nmetamarkers)
    loadings.to_feather(f'{processeddir}/_pcloadings.feather')
    del metapixels, allmp; gc.collect()

    # apply the PC loadings to plain pixels
    return pca_pixels(normeddir, masksdir, loadings, sids)    

def harmonize(allpixels_pca, path_to_Rscript, sid_to_covs=None, plot=True):
    # add covariates
    if sid_to_covs is not None:
        harmony_cov_names = list(sid_to_covs.columns)
    else:
        harmony_cov_names = []
    for cov_name in harmony_cov_names:
        allpixels_pca[cov_name] = allpixels_pca['sid'].map(sid_to_covs[cov_name])
    harmony_cov_names = ['sid'] + harmony_cov_names
    
    # write out the data
    with tempfile.NamedTemporaryFile(delete=False, suffix=".feather") as temp_file:
        allpixels_pca.to_feather(temp_file.name)
    
    # run harmony script
    path_to_script = os.path.dirname(__file__) + '/harmonize.R'
    command = [path_to_Rscript, path_to_script, temp_file.name] + harmony_cov_names
    try:
        print('=== running harmony ===')
        subprocess.run(command, check=True)
        print('=== finished running harmony ===')
    except subprocess.CalledProcessError as e:
        print(f"Error running harmony: {e}")
        return None
    
    # collect output of harmony script and visualize
    base, ext = os.path.splitext(temp_file.name)
    harmpixels = pd.read_feather(f"{base}_harmony{ext}")
    if plot:
        visualize_pixels(harmpixels, 50000, 'harm. metamarkers', harmony_cov_names)
    
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
        s.to_netcdf(f'{processeddir}/{sid}.nc', encoding={s.name: compression}, engine="netcdf4")
        gc.collect()

def sanity_checks(outdir, repname, sid_to_covs=None):
    processeddir = f'{outdir}/{repname}'
    sids = [os.path.splitext(f)[0]
        for f in os.listdir(processeddir) if f.endswith('.nc')]
    if sid_to_covs is not None:
        harmony_cov_names = list(sid_to_covs.columns)
    else:
        harmony_cov_names = []

    print('all PCs of one sample')
    s = xr.open_dataarray(f'{processeddir}/{sids[0]}.nc').astype(np.float32)
    s.plot(col='marker', col_wrap=5, vmin=-10, vmax=10, cmap='seismic')

    print('histogram of each pc')
    harmpixels = pd.read_feather(f'{processeddir}/_allpixels_pca_harmony.feather')
    nmms = harmpixels.values.shape[1] - len(harmony_cov_names) - 1 # the -1 accounts for sid
    plt.figure(figsize=(3*4, 2*int(np.ceil(nmms/4))))
    for i in range(nmms):
        print(i, end='')
        plt.subplot(int(np.ceil(nmms/4)), 4, i+1)
        plt.hist(harmpixels.values[:,i], bins=1000)
    plt.tight_layout()
    plt.show()

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