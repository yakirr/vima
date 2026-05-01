import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from ..data import samples as vds

pb = lambda x: tqdm(x, ncols=100)


def plot_sample_with_patches(s, marker, patchmeta, remove_margin=False, ax=None, show=True, **kwargs):
    if ax is None: ax = plt.gca()

    inpatches = vds.union_patches_in_sample(patchmeta, s).astype(np.uint8)
    
    if remove_margin and inpatches.sum() > 0:
        indices = np.where(inpatches)
        x_min, x_max = np.min(indices[1]), np.max(indices[1])
        y_min, y_max = np.min(indices[0]), np.max(indices[0])
        x_min = max(x_min-200, 0); x_max = min(x_max+200, inpatches.sizes['x'])
        y_min = max(y_min-200, 0); y_max = min(y_max+200, inpatches.sizes['y'])
    else:
        x_min, x_max = 0, inpatches.sizes['x']
        y_min, y_max = 0, inpatches.sizes['y']

    # find contours of the patches in the sample
    contours, _ = cv2.findContours(inpatches.data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # plot
    ax.imshow(s.sel(marker=marker).data, **kwargs, cmap='seismic')
    for cnt in contours:
        cnt = cnt.squeeze()  # remove unnecessary dimensions
        ax.plot(cnt[:, 0], cnt[:, 1], color='black')
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)

    if show:
        plt.show()


def plot_samples_with_patches(samples, marker, patchmeta, ncols=5, **kwargs):
    nrows = int(np.ceil(len(samples) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    for ax, s in pb(zip(axs.flatten(), samples)):
        plot_sample_with_patches(s, marker, patchmeta, ax=ax, show=False, **kwargs)
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
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.show()


def _adjust_resolution(mypatches):
    import math
    from functools import reduce
    mypatches = mypatches.copy()
    if mypatches.patchsize.nunique() > 1:
        raise ValueError('All patches must have the same patchsize')
    stride = reduce(math.gcd, list(mypatches.x.astype(int)) + list(mypatches.y.astype(int)))
    mypatches.x = mypatches.x // stride
    mypatches.y = mypatches.y // stride
    return mypatches

def spatialplot(patchmeta, values, sids=None, cmap='viridis', vmin=None, vmax=None,
                ncols=5, size=3, empty_color='black', show=True):
    import copy
    if sids is None:
        sids = list(patchmeta.sid.unique())
    nrows = int(np.ceil(len(sids) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows),
                            facecolor='black', squeeze=False)
    axs = axs.flatten()

    cmap_obj = copy.copy(plt.get_cmap(cmap))
    cmap_obj.set_bad(empty_color)

    if vmin is None:
        vmin = np.percentile(values, 5)
    if vmax is None:
        vmax = np.percentile(values, 95)

    sid_to_ax = {}
    for i, sid in enumerate(sids):
        ax = axs[i]
        mypatches = _adjust_resolution(patchmeta[patchmeta.sid == sid])
        h = int(mypatches['y'].max())+1
        w = int(mypatches['x'].max())+1

        occupancy = np.zeros((h, w), dtype=bool)
        for x, y in mypatches[['x', 'y']].values.astype(int):
            occupancy[y, x] = True
        row_keep = np.where(occupancy.any(axis=1))[0]
        col_keep = np.where(occupancy.any(axis=0))[0]

        canvas = np.full((h, w), np.nan)
        myvalues = values.reindex(mypatches.index)
        for (x, y), v in zip(mypatches[['x', 'y']].values.astype(int), myvalues):
            canvas[y, x] = v
        canvas = canvas[np.ix_(row_keep, col_keep)]

        ax.imshow(canvas, cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax._vima_row_keep = row_keep
        ax._vima_col_keep = col_keep
        ax.set_title(sid, color='white', fontsize=8)
        ax.set_facecolor('black')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        sid_to_ax[sid] = ax

    for ax in axs[len(sids):]:
        ax.axis('off')

    fig.tight_layout()
    if show:
        plt.show()
    else:
        return sid_to_ax


def annotate_spatialplot(sid_to_ax, patchmeta, highlight, color, thickness=3, show=True):
    import cv2
    for sid, ax in sid_to_ax.items():
        mypatches = _adjust_resolution(patchmeta[patchmeta.sid == sid])
        h = int(mypatches['y'].max())+1
        w = int(mypatches['x'].max())+1
        mask = np.zeros((h, w), dtype=np.uint8)

        flagged = mypatches[highlight.reindex(mypatches.index).fillna(False)]
        for x, y in flagged[['x', 'y']].values.astype(int):
            mask[y,x] = 1

        if mask.max() == 0:
            continue

        row_keep = getattr(ax, '_vima_row_keep', None)
        col_keep = getattr(ax, '_vima_col_keep', None)
        if row_keep is not None:
            mask = mask[np.ix_(row_keep, col_keep)]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            cnt = cnt.squeeze()
            if cnt.ndim == 1:
                continue
            ax.plot(cnt[:, 0], cnt[:, 1], color=color, linewidth=thickness)

    if show:
        plt.show()
    else:
        return sid_to_ax