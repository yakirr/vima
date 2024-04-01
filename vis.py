import matplotlib.pyplot as plt
import numpy as np

def plot(model, epoch, test_sample, sids, show=True, rgb=False, pmin=None, pmax=None):
    predictions = model(test_sample)

    if not rgb:
        fig = plt.figure(figsize=(32,3*4))
        for j, channel in enumerate([0,1,2]):
            vmin = -np.max(np.abs(test_sample[:,:,:,channel]))
            vmax = -vmin
            for i, (a, b) in enumerate(zip(predictions, test_sample)):
                plt.subplot(6, len(test_sample), len(test_sample)*(2*j) + i + 1)
                plt.imshow(b[:,:,channel], vmin=vmin, vmax=vmax, cmap='seismic')
                plt.axis('off')
                plt.subplot(6, len(test_sample), len(test_sample)*(2*j+1) + i + 1)
                plt.imshow(a[:,:,channel], vmin=vmin, vmax=vmax, cmap='seismic')
                plt.axis('off')
    else:
        if pmin is None:
            pmin = np.array([0,0,0])
        if pmax is None:
            pmax = np.array([1,1,1])
        fig = plt.figure(figsize=(2*len(test_sample), 4))
        for i, (a, b) in enumerate(zip(predictions, test_sample)):
            plt.subplot(2, len(test_sample), i+1)
            plt.imshow((b - pmin)/(pmax-pmin))
            plt.axis('off')
            plt.subplot(2, len(test_sample), len(test_sample)+i+1)
            plt.imshow((a - pmin)/(pmax-pmin))
            plt.axis('off')
    plt.tight_layout()

    if show:
        plt.show()