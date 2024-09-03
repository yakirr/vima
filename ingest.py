import numpy as numpy
import pandas as pd
import numpy as np

def transcriptlist_to_pixellist(transcriptlist, x_colname='global_x', y_colname='global_y', gene_colname='gene', pixel_size=10):
    # adds dummy rows such that there is at least one entry for every possible x- and y- value
    # between the min and max values
    def complete(pl, colname, genes, fill=0., verbose=True):
        vals = np.sort(pl[colname].unique())
        min_col = vals.min() // 1
        max_col = vals.max() // 1
        delta = int(min(vals[1:] - vals[:-1]))
        full_range = list(np.arange(min_col, max_col + 1, delta))
        locs_toadd = np.setdiff1d(full_range, vals)
        if verbose: print(f'\tadding {colname}={locs_toadd}')
        toadd = pl.iloc[:len(locs_toadd)].copy()
        toadd[colname] = locs_toadd
        toadd[genes] = fill
        return pd.concat([pl, toadd], axis=0, ignore_index=True)

    transcriptlist = transcriptlist[[x_colname, y_colname, gene_colname]].copy()
    transcriptlist['pixel_x'] = (transcriptlist[x_colname] / pixel_size).astype(int) * pixel_size
    transcriptlist['pixel_y'] = (transcriptlist[y_colname] / pixel_size).astype(int) * pixel_size

    pixels = transcriptlist.groupby(['pixel_x', 'pixel_y'])[gene_colname].value_counts().unstack(fill_value=0)
    pixels.reset_index(inplace=True)
    pl = pixels.rename_axis(None, axis=1)
    genes = pl.columns[2:]

    return complete(complete(pl, 'pixel_x', genes), 'pixel_y', genes)

def pixellist_to_pixelmatrix(pl, markers):
    s = pd.pivot_table(pl, values=markers, index='pixel_y', columns='pixel_x').fillna(0)
    s.columns.names = ['markers', 'pixel_x']
    print('sample shape:', s.shape)
    return s

# mode can be either 'ntranscripts' or 'adaptive'
def get_foreground_st(s, min_ntranscripts=10, plot=True):
    # make mask
    totals = s.sum(levels='markers')
    mask = totals > min_ntranscripts
    print('npixels:', mask.values.sum())
    return mask

def get_pixels(s, mask):
    markers = s.get_level_values('markers').unique()
    return pd.DataFrame(s[mask.values], columns=markers)

#######
    # # plot
    # if plot:
	#     sample = np.zeros((*mask.shape, len(genes)))
	#     sample[mask.values] = pixels.values
	#     totals = sample.sum(axis=2)
	#     plt.imshow(totals, cmap='Reds', vmin=0)
	#     plt.imshow(mask, cmap='gray', vmin=0, vmax=1, alpha=0.5)
	#     plt.axis('off')
	#     plt.show()