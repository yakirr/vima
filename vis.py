import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import scale
import torch

def plot_with_reconstruction(model, examples, show=True, channels=[0,1,2], pmin=None, pmax=None):
    examples = examples.permute(0,3,1,2)
    model.eval()
    with torch.no_grad():
        means, _ = model.encode(examples)
        predictions = model.decode(means).permute(0,2,3,1).cpu().numpy()
    examples = examples.permute(0,2,3,1).cpu().numpy()
    losses = np.mean((examples - predictions)**2, axis=(1,2,3))*examples.shape[1]*examples.shape[2]*examples.shape[3]


    fig = plt.figure(figsize=(32,len(channels)*4))
    for j, channel in enumerate(channels):
        v = max(abs(pmin[channel]), abs(pmax[channel]))
        for i, (a, b) in enumerate(zip(predictions, examples)):
            plt.subplot(2*len(channels), len(examples), len(examples)*(2*j) + i + 1)
            plt.imshow(b[:,:,channel], vmin=-v, vmax=v, cmap='seismic')
            plt.axis('off')
            if j == 0:
                plt.text(20, 1, f'{losses[i]:.2f}', ha='center', va='bottom', fontsize=16)
            plt.subplot(2*len(channels), len(examples), len(examples)*(2*j+1) + i + 1)
            plt.imshow(a[:,:,channel], vmin=-v, vmax=v, cmap='seismic')
            plt.axis('off')

    plt.tight_layout()
    if show:
        plt.show()

def plot_patches_separatechannels(examples, choose=None, vmax=10, vmin=None, channels=[0,1,2], channelnames=None):
    if choose is not None:
        examples = examples[np.random.choice(range(len(examples)), size=min(choose, len(examples)), replace=False)]
    if vmin is None:
        vmin = -vmax
    
    fig = plt.figure(figsize=(len(examples)*1.5, len(channels)*1.5))
    for j, channel in enumerate(channels):
        for i, a in enumerate(examples):
            plt.subplot(len(channels), len(examples), i + j*len(examples) + 1)
            plt.imshow(a[:,:,channel], vmin=vmin, vmax=vmax, cmap='seismic')
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

# colormaps consists of tuples of the form [channel, color, scaler]
def plot_patches_overlaychannels_som(examples, latent, colormaps, nx=5, ny=5, show=True, seed=None):
    if seed is not None: np.random.seed(seed)
    som = MiniSom(nx, ny, len(latent[0]),
              neighborhood_function='gaussian', sigma=1.5,
              random_seed=1)
    latent = scale(latent)
    # latent = latent - latent.mean(axis=0)
    som.pca_weights_init(latent)
    som.train_random(latent, 1000, verbose=False)
    map = som.labels_map(latent, range(len(latent)))
    
    fig, axs = plt.subplots(nx, ny, figsize=(nx,ny))
    for p, box in map.items():
        box = list(box)
        c = np.random.choice(box, size=1)[0]
        image = apply_colormap(examples[c], colormaps)
        ax = axs[p[0],p[1]]
        ax.imshow(image)
    for ax in axs.flatten():
        ax.axis('off')
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