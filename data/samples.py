import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv2
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)

def default_parser(fname):
    fname = fname.split('/')[-1].split('.nc')[0]
    return {
        'donor': fname.split('_')[0],
        'sid': fname.split('_')[1]
    }

def read_samples(filemask, filename_parser, stop_after=None):
    samples = {}
    files = glob.glob(filemask)
    if stop_after is None: stop_after = len(files)

    for f in pb(files[:stop_after]):
        s = xr.open_dataarray(f).astype(np.float32)
        s.attrs.update(filename_parser(f))
        samples[s.sid] = s

    return samples

def get_mask(s):
    return (s!=0).any(dim='marker')

def choose_patches(samples, patchsize, patchstride, max_frac_empty=0.2):
    patch_meta = []

    for s in pb(samples.values()):
        mask = get_mask(s)
        starts = np.array([
            [i, j]
            for i in range(0, mask.sizes['x']-patchsize, patchstride)
            for j in range(0, mask.sizes['y']-patchsize, patchstride)
            if mask.data[j:j+patchsize, i:i+patchsize].mean() > (1-max_frac_empty)
        ]).astype('int')

        patch_meta.append(pd.DataFrame([
                (s.sid, s.donor, i, j, mask.x[i], mask.y[j])
                for i, j in starts
            ],
            columns=['sid','donor','x','y', 'x_microns', 'y_microns'],
        ))
    patch_meta = pd.concat(patch_meta, axis=0).reset_index(drop=True)
    patch_meta.x = patch_meta.x.astype('int')
    patch_meta.y = patch_meta.y.astype('int')
    patch_meta.x_microns = patch_meta.x_microns.astype('float32')
    patch_meta.y_microns = patch_meta.y_microns.astype('float32')
    patch_meta['patchsize'] = patchsize
    return patch_meta

def union_patches_in_sample(patchmeta, s):
    res = s[:,:,0].copy()
    res[:,:] = 0

    for _, p in patchmeta[patchmeta.sid == s.sid].iterrows():
        res[p.y:p.y+p.patchsize, p.x:p.x+p.patchsize] = 1

    return res

def get_boundary(mask, color, thickness=10):
    res = np.zeros((mask.shape[0], mask.shape[1], 4)).astype('float32')
    if np.max(mask) == 0:
        return res
    mask = (mask / np.max(mask) * 255).astype('uint8')
    ret, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(res, contours, -1, color, thickness)
    return res

def plot_sample_with_patches(s, marker, patchmeta, remove_margin=False, ax=None, show=True, **kwargs):
    if ax is None: ax = plt.gca()

    inpatches = union_patches_in_sample(patchmeta, s)
    
    if remove_margin:
        pass
        # indices = np.where(mypatches)
        # x_min, x_max = np.min(indices[0]), np.max(indices[0])
        # y_min, y_max = np.min(indices[1]), np.max(indices[1])
        # x_min = max(x_min-200, 0); x_max = min(x_max+200, image.shape[0])
        # y_min = max(y_min-200, 0); y_max = min(y_max+200, image.shape[1])
    else:
        x_min, x_max = 0, inpatches.sizes['x']
        y_min, y_max = 0, inpatches.sizes['y']
    
    # Display the image with contours
    thickness = int(max(x_max-x_min, y_max-y_min)/100)+1
    s = s[y_min:y_max, x_min:x_max]
    contour = xr.DataArray(
        get_boundary(inpatches.data, (0,0,0,1), thickness=thickness)[y_min:y_max, x_min:x_max],
        dims=['y','x','rgba'],
        coords={'y':s.y, 'x':s.x})

    s.sel(marker=marker).plot(ax=ax, zorder=0, **kwargs)
    contour.plot.imshow(rgb='rgba', ax=ax, zorder=1)
    ax.set_aspect('equal')

    if show:
        plt.show()

def plot_samples_with_patches(samples, marker, patchmeta, ncols=5, **kwargs):
    nrows = int(np.ceil(len(samples) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    for ax, s in pb(zip(axs.flatten(), samples)):
        plot_sample_with_patches(s, marker, patchmeta, ax=ax, show=False, add_colorbar=False, **kwargs)
        ax.set_title(s.sid)
    fig.tight_layout()
    fig.show()

def plot_npatches_per_sample(samples, patchmeta):
    res = patchmeta.sid.value_counts()
    empty = [sid for sid in samples.keys() if sid not in patchmeta.sid.unique()]
    for sid in empty:
        res.loc[sid] = 0

    plt.figure(figsize=(15,2))
    plt.bar(x=res.index, height=res)
    plt.tick_params(axis='x', rotation=90)
    plt.show()