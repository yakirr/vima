import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import cv2
import torch
import xarray as xr
import seaborn as sns
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)
from .data import samples as tds

def plot_association(D, key='mncoef', fdr_thresh=0.1, ax=None, show=True, **kwargs):
    if ax is None:
        ax = plt.gca()
    sig = D.obs[key].where(D.obs[f'{key}_fdr'] <= fdr_thresh, 0)
    D = D.copy()
    D.obs['_sig'] = sig
    sc.pl.umap(D, ax=ax, show=False, **kwargs)
    sc.pl.umap(D[D.obs._sig != 0], color='_sig', cmap='seismic', vmin=-1, vmax=1,
               ax=ax, title=f'{(D.obs._sig != 0).sum()} microniches at FDR {fdr_thresh*100:.0f}%',
               show=False, **kwargs)
    if show:
        plt.show()

def plot_with_reconstruction(model, examples, show=True, channels=[0,1,2], pmin=None, pmax=None, cmap='seismic'):
    examples = (examples[0].permute(0,3,1,2), examples[1])
    model.eval()
    with torch.no_grad():
        predictions, means, _ = model.forward(examples)
    examples = examples[0].permute(0,2,3,1).cpu().numpy()
    predictions = predictions.permute(0,2,3,1).cpu().numpy()
    losses = np.mean((examples - predictions)**2, axis=(1,2,3))

    fig = plt.figure(figsize=(32,len(channels)*4))
    for j, channel in enumerate(channels):
        for i, (a, b) in enumerate(zip(predictions, examples)):
            plt.subplot(2*len(channels), len(examples), len(examples)*(2*j) + i + 1)
            plt.imshow(b[:,:,channel], vmin=pmin[channel], vmax=pmax[channel], cmap=cmap)
            plt.axis('off')
            if j == 0:
                plt.text(20, 1, f'{losses[i]:.2f}', ha='center', va='bottom', fontsize=16)
            plt.subplot(2*len(channels), len(examples), len(examples)*(2*j+1) + i + 1)
            plt.imshow(a[:,:,channel], vmin=pmin[channel], vmax=pmax[channel], cmap=cmap)
            plt.axis('off')

    if show:
        plt.tight_layout()
        plt.show()

def plot_patches_separatechannels(examples, choose=None, vmax=10, vmin=None, channels=[0,1,2], channelnames=None):
    if choose is not None:
        examples = examples[np.random.choice(range(len(examples)), size=min(choose, len(examples)), replace=False)]
    if vmin is None:
        vmin = -vmax
    if isinstance(vmax, (int, float, str)):
        vmax = [vmax] * len(channels)
        vmin = [vmin] * len(channels)
    
    fig = plt.figure(figsize=(len(examples)*1.5, len(channels)*1.5))
    for j, channel in enumerate(channels):
        for i, a in enumerate(examples):
            plt.subplot(len(channels), len(examples), i + j*len(examples) + 1)
            plt.imshow(a[:,:,channel], vmin=vmin[channel], vmax=vmax[channel], cmap='seismic')
            plt.axis('off')
            if channelnames is not None and i == 0:
                plt.gca().text(-5, 20, channelnames[j], va='center', ha='right', rotation=90)

    plt.tight_layout()
    plt.show()

def scaler(minimum=0, maximum=255):
    def rescale(x):
        return np.minimum(np.maximum((x - minimum) / (maximum - minimum), 0), 1)
    return rescale

def apply_colormap(pieces, colormaps):
    if len(pieces.shape) == 3:
        pieces = np.array([pieces])
        reshape = True
    else:
        reshape = False
    images = np.zeros((pieces.shape[0], pieces.shape[1], pieces.shape[2], 3))
    for [channel, color, scaler] in colormaps:
        images += (scaler(pieces[:,:,:,channel])[:,:,:,None] * np.array(color)[None,None,None,:])
    images[images > 1] = 1

    if reshape:
        images = images[0]

    return images

# colormaps consists of tuples of the form [channel, color, scaler]
def plot_patches_overlaychannels(examples, colormaps, nx=5, ny=5, show=True, seed=None):
    if nx*ny < len(examples):
        if seed is not None: np.random.seed(seed)
        ix = np.random.choice(range(len(examples)), size=nx*ny, replace=False)
        examples = examples[ix]
    else:
        ix = range(len(examples))

    images = apply_colormap(examples, colormaps)
    
    fig = plt.figure(figsize=(nx,ny))
    for i, a in enumerate(images):
        plt.subplot(ny,nx,i+1)
        plt.imshow(a)
        plt.axis('off')
    plt.tight_layout()
    if show:
        plt.show()
    return ix

from scipy.optimize import linear_sum_assignment
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
def plot_patches_overlaychannels_linsum(patches, latents, colormaps, nx=5, ny=5, show=True, seed=None, scale_factor=1, spacing=None,
        scalebar=True, scalebar_size=10):
    if nx*ny < len(patches):
        if seed is not None: np.random.seed(seed)
        ix = np.random.choice(len(patches), size=nx*ny, replace=False)
    else:
        ix = range(len(patches))
    patches = patches[ix]
    latents = sc.AnnData(X=latents[ix])
    sc.pp.neighbors(latents, use_rep='X')
    sc.tl.umap(latents)
    gridpoints = np.array([[i,j] for i in range(nx) for j in range(ny)])
    coords = latents.obsm['X_umap']
    coords[:,0] = (coords[:,0] - min(coords[:,0])) / (max(coords[:,0])-min(coords[:,0])) * (nx-1)
    coords[:,1] = (coords[:,1] - min(coords[:,1])) / (max(coords[:,1])-min(coords[:,1])) * (ny-1)

    cost_matrix = np.zeros((len(latents), len(latents)))
    for i, (gx, gy) in enumerate(gridpoints):
        for j, (ex, ey) in enumerate(coords):
            cost_matrix[i, j] = np.linalg.norm([gx - ex, gy - ey])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    final_grid = {j: i for i, j in zip(row_ind, col_ind)}
    
    fig, axs = plt.subplots(nx, ny,
        figsize=(scale_factor*nx,scale_factor*ny))
    for i, patch in enumerate(patches):
        x, y = gridpoints[final_grid[i]]
        ax = axs[int(x), int(y)]
        image = apply_colormap(patch, colormaps)
        ax.imshow(image)
    for ax in axs.flatten():
        ax.axis('off')
        
    if spacing is None:
        plt.tight_layout()
    else:
        spacing *= scale_factor
        plt.subplots_adjust(left=spacing/2, right=1-spacing/2, top=1-spacing/2, bottom=spacing/2, wspace=spacing, hspace=spacing)

    if scalebar:
        scalebar = AnchoredSizeBar(axs[-1,-1].transData,
            scalebar_size, '', 'lower right', pad=0.2, label_top=True, color='white', frameon=False, size_vertical=2,)
        axs[-1,-1].add_artist(scalebar)
    
    if show:
        plt.show()
    else:
        return fig

def plot_patches_overlaychannels_sorted(examples, colormaps, labels=None, nx=5, ny=5, show=True):
    images = apply_colormap(examples, colormaps)
    
    fig, axs = plt.subplots(nx, ny, figsize=(nx,ny))
    for i, a in enumerate(images[:nx*ny]):
        plt.subplot(ny,nx,i+1)
        plt.imshow(a)
        if labels is not None:
            plt.text(2, 10, f'{labels[i]}', color='white')
        plt.axis('off')
    plt.tight_layout()
    if show:
        plt.show()

# each color channel should be a tuple of the form (channel, scaler)
def plot_patches_fourcolors(examples, nx=5, ny=5,
            red=(None, None), cyan=(None, None), green=(None, None), yellow=(None, None), show=True):
    colormaps = []
    
    if red[0] is not None:
        colormaps.append([red[0], [1,0,0], red[1]])
    if green[0] is not None:
        colormaps.append([green[0], [0,1,0], green[1]])
    if cyan[0] is not None:
        colormaps.append([cyan[0], [0,1,1], cyan[1]])
    if yellow[0] is not None:
        colormaps.append([yellow[0], [1,1,0], yellow[1]])

    plot_patches_overlaychannels(examples, colormaps, nx=nx, ny=ny, show=show)

def diff_markers(
    features,
    group_a,
    group_b=None,
    n_top=10,
    n_bottom=0,
    markers=None,
    labels=None,
    kind='violin',
    ax=None,
    show=True,
    **kwargs,
):
    """Plot comparing marker distributions between two patch groups.

    Args:
        features: DataFrame (n_patches × n_markers), e.g. from expression_profiles.
        group_a: boolean array, length n_patches — first group.
        group_b: boolean array or None — second group; defaults to ~group_a.
        n_top: markers most enriched in group_a to show (default 10).
        n_bottom: markers most enriched in group_b to show (default 0).
        markers: explicit list of markers to plot; overrides n_top/n_bottom.
        labels: [label_a, label_b] for the legend; defaults to ['a', 'b'].
        ax: matplotlib axes; defaults to current axes.
        show: call plt.show() when done (default True).
        kind: type of plot to create ('violin' or 'box').
        **kwargs: additional arguments passed to seaborn plotting function.

    Returns:
        List of markers plotted, in display order.
    """
    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = ['a', 'b']

    group_a = np.asarray(group_a, dtype=bool)
    group_b = ~group_a if group_b is None else np.asarray(group_b, dtype=bool)

    diffs = (features[group_a].median() - features[group_b].median()).sort_values(ascending=False)

    if markers is not None:
        toplot = list(markers)
    else:
        toplot = list(diffs.index[:n_top])
        if n_bottom > 0:
            toplot = toplot + list(diffs.index[-n_bottom:])

    mask = group_a | group_b
    df = features.loc[mask, toplot].copy()
    df['status'] = np.where(group_a[mask], labels[0], labels[1])
    df = df.melt(id_vars='status', value_vars=toplot, var_name='marker', value_name='value')

    plot_fn = {'violin': sns.violinplot, 'box': sns.boxplot}.get(kind)
    if plot_fn is None:
        raise ValueError(f'kind must be "violin" or "box"; got {kind!r}')
    plot_fn(data=df, x='marker', y='value', hue='status', order=toplot, ax=ax, **kwargs)
    if show:
        plt.show()

def spatialplot(samples, sortkey, allpatches, scores, rgbs=[[1.,0.,0.]],
        labels=None,
        highlights=None, outline_rgbas=None, outline_thickness=10,
        skipthresh=10, skipevery=1, stopafter=None, label_fontsize=12,
        scalebar=False, scalebar_size=100,
        vmax=1, ncols=5, size=2, filterempty=False, show=True):
    toplot = allpatches[allpatches.sid.isin(samples.keys())].sid.value_counts() > skipthresh
    nsamples = len(sortkey[toplot].sort_values().index[::skipevery])
    nrows = int(np.ceil(nsamples/ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(ncols*size,nrows*size))

    for ax, sid in pb(zip(axs.flatten(), sortkey[toplot].sort_values().index[::skipevery])):
        mypatches = allpatches[allpatches.sid == sid]

        canvas = tds.union_patches_in_sample(mypatches, samples[sid])
        if filterempty:
            indices = np.where(canvas)
            nonempty_rows = canvas.sum(axis=1) > 0
            nonempty_cols = canvas.sum(axis=0) > 0
        else:
            nonempty_rows = range(len(canvas))
            nonempty_cols = range(len(canvas[0]))

        ax.imshow(canvas[nonempty_rows][:,nonempty_cols], cmap='grey')
        for score, color in zip(scores, rgbs):
            sigcanvas = np.zeros((*canvas.shape, 4))
            sigcanvas[:,:,:3] = color
            
            score_ = score[mypatches.index].values / vmax
            for (x,y,ps), s in zip(mypatches[['x','y','patchsize']].values, score_):
                x,y,ps = int(x), int(y), int(ps)
                sigcanvas[y:y+ps,x:x+ps,-1] += s
            sigcanvas[sigcanvas > 1] = 1
            ax.imshow(sigcanvas[nonempty_rows][:,nonempty_cols])

        if highlights is not None:
            for highlight, outline_rgba in zip(highlights, outline_rgbas):
                myhighlight = highlight[mypatches.index]
                mask = tds.union_patches_in_sample(mypatches[myhighlight != 0], samples[sid])
                boundary = tds.get_boundary(mask.data, outline_rgba, thickness=outline_thickness)
                ax.imshow(boundary[nonempty_rows][:,nonempty_cols])
        if labels is not None:
            ax.set_title(labels[sid], color='white', fontsize=label_fontsize)

        if scalebar:
            scalebar = AnchoredSizeBar(ax.transData,
                scalebar_size, '', 'lower right', pad=0.2, label_top=True, color='white', frameon=False, size_vertical=2,)
            ax.add_artist(scalebar)

        if stopafter is not None and ax == axs.flatten()[stopafter-1]:
            break

    for ax in axs.flatten()[nsamples:]:
        ax.imshow(np.zeros((10,10)), cmap='grey', vmin=0, vmax=1)

    for ax in axs.flatten():
        ax.spines[['top','bottom','left','right']].set_visible(False) # can also do set_color('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_tick_params(length=0)
        ax.yaxis.set_tick_params(length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

            
    fig.patch.set_facecolor('black')

    if show:
        plt.show()
    else:
        return fig