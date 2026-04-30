import matplotlib.pyplot as plt
import numpy as np
import torch
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


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


def plot_patches_overlaychannels_linsum(patches, latents, colormaps, nx=5, ny=5, show=True, seed=None,
        scale_factor=1, spacing=None, scalebar=True, scalebar_size=10):
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

    fig, axs = plt.subplots(nx, ny, figsize=(scale_factor*nx, scale_factor*ny))
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
        plt.subplots_adjust(left=spacing/2, right=1-spacing/2, top=1-spacing/2, bottom=spacing/2,
                            wspace=spacing, hspace=spacing)

    if scalebar:
        scalebar = AnchoredSizeBar(axs[-1,-1].transData,
            scalebar_size, '', 'lower right', pad=0.2, label_top=True, color='white', frameon=False, size_vertical=2)
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
