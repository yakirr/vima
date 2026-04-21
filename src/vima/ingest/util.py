
import numpy as np
import pandas as pd
import xarray as xr
import gc
import matplotlib.pyplot as plt

compression = {'zlib': True, 'complevel': 2} # settings for writing xarrays

###########################################
# utility functions
###########################################
def xr_to_pixellist(s, mask):
    return s.data[mask.data]

def set_pixels(s, mask, pl):
    s.data[mask.data] = pl

def ar():
    plt.gca().set_aspect('equal')

def write_xarray(s, fname):
    s.to_netcdf(fname, encoding={s.name: compression}, engine="netcdf4")

###########################################
# for creating raw pixel files
###########################################
def transcriptlist_to_pixellist(transcriptlist, x_col, y_col, gene_col, pixel_size=10, verbose=False):
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

    transcriptlist = transcriptlist[[x_col, y_col, gene_col]].copy()
    transcriptlist['pixel_x'] = (transcriptlist[x_col] / pixel_size).astype(int) * pixel_size
    transcriptlist['pixel_y'] = (transcriptlist[y_col] / pixel_size).astype(int) * pixel_size

    pixels = transcriptlist.groupby(['pixel_x', 'pixel_y'])[gene_col].value_counts().unstack(fill_value=0)
    pixels.reset_index(inplace=True)
    pl = pixels.rename_axis(None, axis=1)
    genes = pl.columns[2:]

    return complete(complete(pl, 'pixel_x', genes, verbose=verbose), 'pixel_y', genes, verbose=verbose)

def df_to_xarray32(df):
    markers = df.columns.get_level_values('markers').unique()
    return xr.DataArray(
            df.values.reshape((len(df), len(markers), -1)).transpose(0,2,1),
            coords={'x': df.columns.get_level_values('pixel_x').unique().values, 'y': df.index.values, 'marker': markers.values},
            dims=['y', 'x', 'marker']
        ).astype(np.float32)

def pixellist_to_pixelmatrix(pl, markers):
    # pivot in pandas
    s = pd.pivot_table(pl, values=markers, index='pixel_y', columns='pixel_x').fillna(0)
    s.columns.names = ['markers', 'pixel_x']

    # convert to xarray
    return df_to_xarray32(s)

def downsample(sample, factor, aggregate=np.mean):
    pad_width = (
        (int(factor - sample.shape[0] % factor), 0),
        (int(factor - sample.shape[1] % factor), 0),
        (0,0))
    sample = np.pad(sample, pad_width, mode='constant', constant_values=0)
    smaller = sample.reshape(sample.shape[0], sample.shape[1]//factor, factor, sample.shape[2])
    smaller = aggregate(smaller, axis=2)
    smaller = smaller.reshape(smaller.shape[0]//factor, factor, smaller.shape[1], smaller.shape[2])
    smaller = aggregate(smaller, axis=1)
    return smaller

def hiresarray_to_downsampledxarray(sample, name, factor, pixelsize, markers):
    sample = downsample(sample, factor)
    sample = xr.DataArray(
            sample,
            coords={'x': np.arange(sample.shape[1])*factor*pixelsize, 'y': np.arange(sample.shape[0])*factor*pixelsize, 'marker': markers},
            dims=['y', 'x', 'marker']
        ).astype(np.float32)
    sample.name = name
    return sample