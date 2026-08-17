import numpy as np
import xarray as xr
from scipy.ndimage import convolve
import scanpy as sc
import anndata as ad
from matplotlib import pyplot as plt
import pandas as pd
from . import util
from .._settings import settings, logger
import gc

###########################################
# dimensionality reduction and integration
###########################################
def metapixels_allsamples(normedpixelsdir, masksdir, sids, total_n_metapixels):
    """
    Pool metapixels across all samples for a more robust PCA fit.

    Loads each sample's normalized pixels, standardizes them with the stored
    per-marker means/stds, builds metapixels, and randomly downsamples each
    sample to roughly ``total_n_metapixels // len(sids)`` metapixels. Warns if a
    sample's markers differ from the first sample's.

    Parameters
    ----------
    total_n_metapixels
        Target total number of metapixels pooled across samples.

    Returns
    -------
    tuple
        ``(all_metapixels, all_npixels)``: dicts mapping sample ID to the
        sample's metapixel DataFrame and to the per-metapixel count of
        contributing non-empty pixels.
    """
    def cdf(v, ax):
        sorted_data = np.sort(v)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf)

    all_metapixels = {}
    all_npixels = {}

    # figure out how many metapixels to store per sample
    nsamples = len(sids)
    nmp_per_sample = total_n_metapixels // nsamples

    if settings.show_plots():
        fig = plt.figure(figsize=(7,5))

    logger.info('Creating metapixels prior to PCA')
    logger.info(f'\t(will randomly downsample to {nmp_per_sample} metapixels per sample if needed.)')
    ref_markers = None
    ref_sid = None
    for i, sid in enumerate(settings.progress(sids, name='creating metapixels')):
        da = xr.open_dataarray(f'{normedpixelsdir}/{sid}.nc')
        mask_da = xr.open_dataarray(f'{masksdir}/{sid}.nc')
        
        # ensure same markers in same order in all files
        markers = list(da.marker.values)
        if ref_markers is None:
            ref_markers, ref_sid = markers, sid
        elif markers != ref_markers:
            missing = set(ref_markers) - set(markers)
            extra   = set(markers) - set(ref_markers)
            order_mismatch = not missing and not extra
            detail = (f'order differs' if order_mismatch else
                      f'{len(missing)} missing, {len(extra)} extra vs {ref_sid}')
            logger.warning(f'{sid} has different markers ({len(markers)}) '
                           f'than {ref_sid} ({len(ref_markers)}): {detail}')
        
        means = xr.DataArray(da.attrs['means'], dims='marker')
        stds = xr.DataArray(da.attrs['stds'], dims='marker')
        da = ((da - means) / stds).where(mask_da, 0)

        all_metapixels[sid], all_npixels[sid] = metapixels(da, mask_da)
        da.close(); mask_da.close()
        del da, mask_da
        if len(all_metapixels[sid]) > nmp_per_sample:
            ix = np.random.choice(len(all_metapixels[sid]), nmp_per_sample, replace=False)
            all_metapixels[sid] = all_metapixels[sid].iloc[ix]
            all_npixels[sid] = all_npixels[sid][ix]

        # visualize distribution of num non-empty pixels per metapixel in this sample
        if settings.show_plots():
            cdf(all_npixels[sid], plt.gca())
        gc.collect()

    if settings.show_plots():
        plt.title('CDF of # non-empty pixels per metapixel, by sample')
        plt.xlabel('# non-empty pixels per metapixel')
        plt.ylabel('Frequency')
        plt.tight_layout()
        settings.show('metapixel_occupancy_cdf')

    return all_metapixels, all_npixels

def metapixels(s, mask, npixels_thresh=0):
    """
    Pool each pixel with its neighbors into a metapixel.

    Sums each marker over a 5x5 window centered on every pixel and divides by
    the number of non-empty (masked) pixels contributing to that window, so each
    metapixel is the average over its non-empty neighbors. Metapixels with at
    most ``npixels_thresh`` contributing pixels are dropped.

    Returns
    -------
    tuple
        ``(metapixel_df, npixels)``: a marker-columned DataFrame of retained
        metapixels and the per-metapixel count of contributing non-empty pixels.
    """
    markers = s.marker.values

    # make metapixels and compute how many non-empty pixels and transcripts are in each metapixel
    kernel = np.ones((5, 5), np.float32)
    mp = convolve(s.data, kernel[:, :, None], mode="constant")
    npixels = convolve(mask.data.astype('float32'), kernel, mode="constant")

    # filter out metapixels with few non-empty pixels
    metapixels_mask = npixels > npixels_thresh

    # divide each metapixel by the # of non-empty pixels that contributed to it and return
    return pd.DataFrame(data=mp[metapixels_mask] / npixels[metapixels_mask][:,None], columns=markers), npixels[metapixels_mask]

# mps should be an array of dataframes containing metapixels
def pca_metapixels(mps, k):
    """
    Fit a ``k``-component PCA on the pooled metapixels.

    Concatenates the per-sample metapixel tables, standardizes each feature,
    fits PCA, and reports the top/bottom features per component.

    Parameters
    ----------
    mps
        Iterable of per-sample metapixel DataFrames (as returned by
        `metapixels_allsamples`).
    k
        Number of principal components to fit.

    Returns
    -------
    tuple
        ``(loadings, C, allmp)``: the gene-by-component loading matrix, the
        feature correlation matrix, and the standardized metapixel AnnData.
    """
    logger.info('merging and standardizing metapixels')
    allmp = pd.concat(mps)
    allmp -= allmp.values.mean(axis=0, dtype=np.float64)
    allmp /= allmp.values.std(axis=0, dtype=np.float64)
    allmp = allmp.fillna(0)
    allmp.index = np.arange(len(allmp)).astype(str)
    allmp = ad.AnnData(X=allmp)
    C = np.corrcoef(allmp.X[::max(1,(len(allmp)//50000))].T)
    logger.info(f'Metapixel matrix: {allmp.shape[0]:,} pixels × {allmp.shape[1]} features')

    logger.info('performing PCA...')
    sc.tl.pca(allmp, n_comps=k)
    loadings = pd.DataFrame(data=allmp.varm['PCs'], columns=[f'PC{i}' for i in range(1,k+1)], index=allmp.var_names)

    logger.info('top/bottom features per PC (features with negative loadings preceded by "-"):')
    top_bottom = {}
    for pc in loadings.columns:
        col = loadings[pc].sort_values(ascending=False)
        if len(col) < 10:
            top_bottom[pc] = [f'-{g}' if col[g] < 0 else g for g in col.index]
        else:
            top_bottom[pc] = list(col.index[:5]) + [f'-{g}' for g in col.index[-5:]]
    logger.info(pd.DataFrame(top_bottom).to_string(index=False))

    if settings.show_plots():
        plt.figure(figsize=(4, len(loadings)/6))
        plt.imshow(loadings, cmap='seismic', vmin=-0.5, vmax=0.5)
        plt.yticks(range(len(loadings)), loadings.index)
        plt.xticks(range(len(loadings.columns)), loadings.columns, rotation=90)
        settings.show('pc_loadings')

    return loadings, C, allmp

def pca_pixels(normedpixelsdir, masksdir, pcloadings, sids):
    """
    Project every sample's pixels onto the given PC loadings.

    For each sample, standardizes the normalized pixels with the stored
    per-marker means/stds, restricts to masked (non-empty) pixels, and projects
    them onto ``pcloadings``.

    Returns
    -------
    DataFrame
        One row per pixel with a column per principal component plus a 'sid'
        column giving the source sample.
    """
    pcs = []
    sid_labels = []

    logger.info('Applying PCA projection to each sample')
    for sid in settings.progress(sids, name='pixels -> PCA space'):
        da = xr.open_dataarray(f'{normedpixelsdir}/{sid}.nc')
        mask_da = xr.open_dataarray(f'{masksdir}/{sid}.nc')
        
        means = xr.DataArray(da.attrs['means'], dims='marker')
        stds = xr.DataArray(da.attrs['stds'], dims='marker')
        da = ((da - means) / stds).where(mask_da, 0)

        # load raw arrays and close before dtype conversion so we never hold
        # two full (H × W × n_genes) copies simultaneously
        data = da.values
        mask = mask_da.values
        da.close(); mask_da.close()
        del da, mask_da; gc.collect()
        pl = data.astype(np.float32, copy=False)[mask]
        del data, mask; gc.collect()

        pl_pca = pl.dot(pcloadings)
        pcs.append(pl_pca)
        sid_labels.append(np.full(pl_pca.shape[0], sid, dtype=object))
        del pl; gc.collect()

    # concatenate
    pcs = np.vstack(pcs)
    sid_labels = np.concatenate(sid_labels)

    allpixels_pca = pd.DataFrame(
        pcs,
        columns=[f'PC{i}' for i in range(1, pcloadings.shape[1] + 1)]
    )
    allpixels_pca['sid'] = sid_labels

    return allpixels_pca
