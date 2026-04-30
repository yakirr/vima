import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from ..data import samples as tds

pb = lambda x: tqdm(x, ncols=100)


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
                scalebar_size, '', 'lower right', pad=0.2, label_top=True, color='white', frameon=False, size_vertical=2)
            ax.add_artist(scalebar)

        if stopafter is not None and ax == axs.flatten()[stopafter-1]:
            break

    for ax in axs.flatten()[nsamples:]:
        ax.imshow(np.zeros((10,10)), cmap='grey', vmin=0, vmax=1)

    for ax in axs.flatten():
        ax.spines[['top','bottom','left','right']].set_visible(False)
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
