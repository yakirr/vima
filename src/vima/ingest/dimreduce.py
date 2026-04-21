import numpy as np
import xarray as xr
from scipy.ndimage import convolve
import scanpy as sc
import anndata as ad
from matplotlib import pyplot as plt
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)
import pandas as pd
from . import util
import gc

###########################################
# dimensionality reduction and integration
###########################################
def metapixels_allsamples(normedpixelsdir, masksdir, sids, plot=True, ncols=8, total_n_metapixels=2_000_000):
    def cdf(v, ax):
        sorted_data = np.sort(v)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf)

    all_metapixels = {}
    all_npixels = {}

    # figure out how many metapixels to store per sample
    nsamples = len(sids)
    nmp_per_sample = total_n_metapixels // nsamples
    

    if plot:
        fig = plt.figure(figsize=(7,5))

    print('Creating metapixels prior to PCA')
    print(f'\t(will randomly downsample to {nmp_per_sample} metapixels per sample if needed.)')
    for i, sid in enumerate(pb(sids)):
        all_metapixels[sid], all_npixels[sid] = metapixels(
            xr.open_dataarray(f'{normedpixelsdir}/{sid}.nc').astype(np.float32),
            xr.open_dataarray(f'{masksdir}/{sid}.nc'))
        if len(all_metapixels[sid]) > nmp_per_sample:
            ix = np.random.choice(len(all_metapixels[sid]), nmp_per_sample, replace=False)
            all_metapixels[sid] = all_metapixels[sid].iloc[ix]
            all_npixels[sid] = all_npixels[sid][ix]

        # visualize distribution of num non-empty pixels per metapixel in this sample
        if plot:
            cdf(all_npixels[sid], plt.gca())
        gc.collect()

    if plot:
        plt.title('CDF of # non-empty pixels per metapixel, by sample')
        plt.xlabel('# non-empty pixels per metapixel')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    return all_metapixels, all_npixels

def metapixels(s, mask, npixels_thresh=0):
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
def pca_metapixels(mps, k, plot=True):
    print('merging and standardizing metapixels')
    allmp = pd.concat(mps)
    allmp -= allmp.values.mean(axis=0, dtype=np.float64)
    allmp /= allmp.values.std(axis=0, dtype=np.float64)
    allmp = allmp.fillna(0)
    allmp.index = np.arange(len(allmp)).astype(str)
    allmp = ad.AnnData(X=allmp)
    C = np.corrcoef(allmp.X[::max(1,(len(allmp)//50000))].T)
    print(f'Metapixel matrix: {allmp.shape[0]:,} pixels × {allmp.shape[1]} features')

    print('performing PCA...')
    sc.tl.pca(allmp, n_comps=k)
    loadings = pd.DataFrame(data=allmp.varm['PCs'], columns=[f'PC{i}' for i in range(1,k+1)], index=allmp.var_names)

    print()
    print('top/bottom features per PC (features with negative loadings preceded by "-"):')
    top_bottom = {}
    for pc in loadings.columns:
        col = loadings[pc].sort_values()
        top_bottom[pc] = list(col.index[-5:]) + [f'-{g}' for g in col.index[:5]]
    s = pd.DataFrame(top_bottom).to_string(index=False).split('\n')
    print('\033[1;36m' + s[0] + '\033[0m')
    print('\n'.join(s[1:]))
    print()

    if plot:
        plt.figure(figsize=(4, len(loadings)/6))
        plt.imshow(loadings, cmap='seismic', vmin=-0.5, vmax=0.5)
        plt.yticks(range(len(loadings)), loadings.index)
        plt.show()

    return loadings, C, allmp

def pca_pixels(normedpixelsdir, masksdir, pcloadings, sids):
    pcs = []
    sid_labels = []

    print('Applying PCA projection to each sample')
    for sid in pb(sids):
        pl = util.xr_to_pixellist(
            xr.open_dataarray(f'{normedpixelsdir}/{sid}.nc').astype(np.float32),
            xr.open_dataarray(f'{masksdir}/{sid}.nc')
        )
        pl_pca = pl.dot(pcloadings)

        pcs.append(pl_pca)
        sid_labels.append(np.full(pl_pca.shape[0], sid, dtype=object))
        del pl, pl_pca; gc.collect()

    # concatenate
    pcs = np.vstack(pcs)
    sid_labels = np.concatenate(sid_labels)

    allpixels_pca = pd.DataFrame(
        pcs,
        columns=[f'PC{i}' for i in range(1, pcloadings.shape[1] + 1)]
    )
    allpixels_pca['sid'] = sid_labels

    return allpixels_pca
