import matplotlib.pyplot as plt
import numpy as np
import torch

def plot(model, examples, show=True, rgb=False, channels=None, pmin=None, pmax=None):
    examples = examples.permute(0,3,1,2)
    model.eval()
    with torch.no_grad():
        means, _ = model.encode(examples)
        predictions = model.decode(means).permute(0,2,3,1).cpu().numpy()
    examples = examples.permute(0,2,3,1).cpu().numpy()
    losses = np.mean((examples - predictions)**2, axis=(1,2,3))*examples.shape[1]*examples.shape[2]*examples.shape[3]

    if not rgb:
        if channels is None:
            channels = [0,1,2]
        fig = plt.figure(figsize=(32,3*4))
        for j, channel in enumerate(channels):
            v = max(abs(pmin[channel]), abs(pmax[channel]))
            for i, (a, b) in enumerate(zip(predictions, examples)):
                plt.subplot(6, len(examples), len(examples)*(2*j) + i + 1)
                plt.imshow(b[:,:,channel], vmin=-v, vmax=v, cmap='seismic')
                plt.axis('off')
                if j == 0:
                    plt.text(20, 1, f'{losses[i]:.2f}', ha='center', va='bottom', fontsize=16)
                plt.subplot(6, len(examples), len(examples)*(2*j+1) + i + 1)
                plt.imshow(a[:,:,channel], vmin=-v, vmax=v, cmap='seismic')
                plt.axis('off')
    else:
        if pmin is None:
            pmin = np.array([0,0,0])
        if pmax is None:
            pmax = np.array([1,1,1])
        fig = plt.figure(figsize=(2*len(examples), 4))
        for i, (a, b) in enumerate(zip(predictions, examples)):
            plt.subplot(2, len(examples), i+1)
            plt.imshow((b - pmin)/(pmax-pmin))
            plt.axis('off')
            plt.subplot(2, len(examples), len(examples)+i+1)
            plt.imshow((a - pmin)/(pmax-pmin))
            plt.axis('off')
    plt.tight_layout()

    if show:
        plt.show()