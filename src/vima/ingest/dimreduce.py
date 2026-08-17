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

    Loads each sample's normalized pixels, builds roughly
    ``total_n_metapixels // len(sids)`` randomly chosen metapixels from it, and
    standardizes them with the stored per-marker means/stds. Warns if a sample's
    markers differ from the first sample's.

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

        means, stds = da.attrs['means'], da.attrs['stds']

        # metapixels are built from the un-standardized pixels and standardized
        # afterward: averaging over a window and the affine (x-mean)/std commute,
        # so this is identical to standardizing the full array first but avoids
        # materializing several (y, x, marker)-sized temporaries.
        mp, all_npixels[sid] = metapixels(da, mask_da, n_metapixels=nmp_per_sample)
        da.close(); mask_da.close()
        del da, mask_da

        mp -= means
        mp /= stds
        all_metapixels[sid] = pd.DataFrame(data=mp, columns=markers)
        del mp

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

def metapixels(s, mask, npixels_thresh=0, n_metapixels=None, window=5):
    """
    Pool each pixel with its neighbors into a metapixel.

    Averages each marker over a ``window``-by-``window`` window centered on a
    pixel, using only the non-empty (masked) pixels in that window, so each
    metapixel is the average over its non-empty neighbors. Metapixels with at
    most ``npixels_thresh`` contributing pixels are dropped.

    Parameters
    ----------
    n_metapixels
        If given, uniformly sample at most this many metapixel centers and
        compute only those. Since the caller typically keeps a small random
        subset anyway, this avoids convolving the whole (y, x, marker) array,
        which dominates the cost for large marker panels.

    Returns
    -------
    tuple
        ``(metapixels, npixels)``: a ``(metapixel, marker)`` float32 array and
        the per-metapixel count of contributing non-empty pixels.
    """
    mask = mask.data
    H, W = mask.shape

    # how many non-empty pixels contribute to each candidate metapixel (cheap: 2D only)
    kernel = np.ones((window, window), np.float32)
    npixels = convolve(mask.astype(np.float32), kernel, mode="constant")

    # pick the metapixel centers, sampling before doing any work over markers
    centers = np.flatnonzero(npixels.ravel() > npixels_thresh)
    if n_metapixels is not None and len(centers) > n_metapixels:
        centers = centers[np.random.choice(len(centers), n_metapixels, replace=False)]
    npixels = npixels.ravel()[centers]

    # sum each window by gathering its non-empty pixels, one neighbor offset at a time
    data = s.data.reshape(H * W, -1)
    mask = mask.ravel()
    r, c = np.divmod(centers, W)
    mp = np.zeros((len(centers), data.shape[1]), np.float32)
    rad = window // 2
    for dr in range(-rad, rad + 1):
        rr = r + dr
        for dc in range(-rad, rad + 1):
            cc = c + dc
            neighbor = rr * W + cc
            contributes = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            contributes &= mask[np.where(contributes, neighbor, 0)]
            i = np.flatnonzero(contributes)
            mp[i] += data[neighbor[i]]

    # divide each metapixel by the # of non-empty pixels that contributed to it and return
    mp /= npixels[:, None]
    return mp, npixels

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
        ``(loadings, allmp)``: the gene-by-component loading matrix and the
        standardized metapixel AnnData.
    """
    logger.info('merging and standardizing metapixels')
    mps = list(mps)
    markers = mps[0].columns
    n = sum(len(mp) for mp in mps)

    # merge into one preallocated float32 matrix and standardize it in place,
    # accumulating the moments in float64. Doing this in pandas instead upcasts
    # the whole matrix to float64 and copies it once per operation, which at
    # these sizes costs more than the PCA itself.
    allmp = np.empty((n, len(markers)), np.float32)
    i = 0
    for mp in mps:
        allmp[i:i+len(mp)] = mp.to_numpy(np.float32, copy=False)
        i += len(mp)
    del mps

    allmp -= (allmp.sum(axis=0, dtype=np.float64) / n).astype(np.float32)
    stds = np.sqrt(np.einsum('ij,ij->j', allmp, allmp, dtype=np.float64) / n)
    stds[stds == 0] = 1  # constant features stay exactly 0, as the old fillna(0) left them
    allmp /= stds.astype(np.float32)

    allmp = ad.AnnData(X=allmp,
                       obs=pd.DataFrame(index=np.arange(n).astype(str)),
                       var=pd.DataFrame(index=markers))
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

    return loadings, allmp

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
    sid_codes = []
    # project in float32: a DataFrame (or float64) right-hand side silently
    # promotes the result, doubling both the projection cost and the size of
    # the returned table, which has a row per pixel in the whole dataset
    loadings = np.ascontiguousarray(np.asarray(pcloadings), dtype=np.float32)

    logger.info('Applying PCA projection to each sample')
    for code, sid in enumerate(settings.progress(sids, name='pixels -> PCA space')):
        da = xr.open_dataarray(f'{normedpixelsdir}/{sid}.nc')
        mask_da = xr.open_dataarray(f'{masksdir}/{sid}.nc')
        
        means, stds = da.attrs['means'], da.attrs['stds']

        # load raw arrays and close before dtype conversion so we never hold
        # two full (H × W × n_genes) copies simultaneously
        data = da.values
        mask = mask_da.values
        da.close(); mask_da.close()
        del da, mask_da; gc.collect()
        pl = data.astype(np.float32, copy=False)[mask]
        del data, mask; gc.collect()

        # standardize the non-empty pixels only, rather than the full
        # (y, x, marker) array; empty pixels are dropped by the mask anyway
        pl -= means
        pl /= stds

        pl_pca = pl.dot(loadings)
        pcs.append(pl_pca)
        sid_codes.append(np.full(pl_pca.shape[0], code, dtype=np.int32))
        del pl; gc.collect()

    # concatenate
    pcs = np.vstack(pcs)
    sid_codes = np.concatenate(sid_codes)

    allpixels_pca = pd.DataFrame(
        pcs,
        columns=[f'PC{i}' for i in range(1, loadings.shape[1] + 1)]
    )
    # categorical rather than an object column: one code per pixel instead of
    # one pointer, over tens of millions of rows
    # drop categories for samples that contributed no pixels, so downstream
    # get_dummies (Harmony) never sees an all-zero batch column
    allpixels_pca['sid'] = pd.Categorical.from_codes(
        sid_codes, categories=list(sids)).remove_unused_categories()

    return allpixels_pca
