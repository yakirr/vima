
import numpy as np
import pandas as pd
import xarray as xr
import scipy.sparse as sp
import gc
from typing import NamedTuple
import matplotlib.pyplot as plt

compression = {'zlib': True, 'complevel': 2} # settings for writing xarrays

###########################################
# utility functions
###########################################
def xr_to_pixellist(s, mask):
    """Extract the masked (non-empty) pixels of ``s`` as a flat ``(pixel, marker)`` array."""
    return s.data[mask.data]

def set_pixels(s, mask, pl):
    """Write pixel values ``pl`` back into the masked positions of ``s`` in place."""
    s.data[mask.data] = pl

def ar():
    """Set the current matplotlib axis to equal aspect ratio."""
    plt.gca().set_aspect('equal')

def write_xarray(s, fname):
    """Write a DataArray to a compressed NetCDF file."""
    s.to_netcdf(fname, encoding={s.name: compression}, engine="netcdf4")

###########################################
# summary statistics
###########################################
def pool_moments(means_df, stds_df, npixels):
    """
    Pool per-sample means and stds into dataset-wide moments.

    Weights each sample by its number of pixels and combines the average
    within-sample variance with the between-sample variance of the means.

    Parameters
    ----------
    means_df, stds_df
        Marker-by-sample DataFrames of each sample's per-marker mean and
        standard deviation, sharing an index.
    npixels
        Number of pixels contributing to each sample's moments, in the same
        order as the columns of ``means_df``.

    Returns
    -------
    tuple
        ``(grand_mean, grand_std)``: per-marker Series indexed like ``means_df``.
    """
    w = np.array(npixels, dtype=np.float64)
    W = w.sum()

    grand_mean   = np.sum((means_df * w).values, axis=1, dtype=np.float64) / W
    mean_of_vars = np.sum((stds_df ** 2 * w).values, axis=1, dtype=np.float64) / W
    var_of_means = ((means_df.subtract(grand_mean, axis=0).values.astype(np.float64) ** 2) * w).sum(axis=1) / W

    return (pd.Series(grand_mean, index=means_df.index),
            pd.Series(np.sqrt(mean_of_vars + var_of_means), index=means_df.index))

###########################################
# for creating raw pixel files
###########################################
class SparsePixels(NamedTuple):
    """
    Transcripts binned to a pixel grid, as a sparse pixel-by-gene count matrix.

    ``counts`` spans the sample's full bounding rectangle -- one row per grid
    position, x-major, so row ``i * len(y) + j`` holds pixel ``(x[i], y[j])``.
    Positions with no transcripts are empty rows rather than stored zeros, which
    is what keeps a large panel tractable: at Xenium 5K densities the dense
    equivalent is ~95-99% zeros and tens of GB per sample.
    """

    counts: sp.csr_matrix   # (len(x) * len(y), len(genes)), float32
    genes: pd.Index         # gene named by each column
    x: np.ndarray           # x coordinate of each grid column, ascending
    y: np.ndarray           # y coordinate of each grid row, ascending

    def coords_of(self, rows):
        """The ``(x, y)`` coordinates of the given rows of ``counts``."""
        return self.x[rows // len(self.y)], self.y[rows % len(self.y)]

    def nonempty(self):
        """Row indices of the pixels holding at least one transcript."""
        return np.flatnonzero(np.diff(self.counts.indptr) > 0)


def _gene_codes(genes):
    """
    Per-transcript integer codes plus the gene index those codes point into.

    Reproduces the column set that ``groupby(...).value_counts().unstack()``
    used to produce: a categorical keeps its declared categories -- including
    unobserved ones, as all-zero columns -- in category order, while any other
    dtype contributes its observed values in sorted order. Transcripts with a
    missing gene get code ``-1``; the caller drops them, as ``value_counts`` did.
    """
    if isinstance(genes.dtype, pd.CategoricalDtype):
        return genes.cat.codes.to_numpy(), pd.Index(genes.cat.categories)
    codes, uniques = pd.factorize(genes.to_numpy(), sort=True)
    return codes, pd.Index(uniques)


def transcriptlist_to_sparsepixels(transcriptlist, x_col, y_col, gene_col, pixel_size=10):
    """
    Bin a transcript table into a sparse per-pixel gene-count table.

    Snaps each transcript to a ``pixel_size`` grid and counts transcripts per
    gene per pixel. The grid covers every coordinate between the min and max in
    each axis, so it is regular by construction and needs no padding rows.
    """
    ix = (transcriptlist[x_col].to_numpy() / pixel_size).astype(int)
    iy = (transcriptlist[y_col].to_numpy() / pixel_size).astype(int)
    codes, genes = _gene_codes(transcriptlist[gene_col])

    x0, x1, y0, y1 = ix.min(), ix.max(), iy.min(), iy.max()
    nx, ny = int(x1 - x0) + 1, int(y1 - y0) + 1
    rows = (ix.astype(np.int64) - x0) * ny + (iy - y0)
    del ix, iy; gc.collect()

    if (codes < 0).any():   # transcripts with no gene assigned
        keep = codes >= 0
        rows, codes = rows[keep], codes[keep]

    # duplicate (pixel, gene) pairs are summed by the coo -> csr conversion
    counts = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, codes)),
                           shape=(nx * ny, len(genes)))
    return SparsePixels(counts, genes,
                        np.arange(x0, x1 + 1) * pixel_size,
                        np.arange(y0, y1 + 1) * pixel_size)


def qc_pixels(counts, min_ntranscripts_per_pixel, min_ngenes_per_pixel):
    """Mask of the pixels passing QC, plus every pixel's total transcript count."""
    totals = np.asarray(counts.sum(axis=1)).ravel()
    ngenes = np.diff(counts.indptr)     # no explicit zeros are ever stored
    keep = (totals >= min_ntranscripts_per_pixel) & (ngenes >= min_ngenes_per_pixel)
    return keep, totals


def lognormalize(counts, totals, target_sum):
    """
    Total-count normalize each pixel to ``target_sum``, then ``log1p``.

    Only the stored entries are touched, since ``log1p(0) == 0`` leaves empty
    pixels empty. Arithmetic is float64 in the same order as the pandas
    expression this replaced (divide by the pixel total, then scale), so the
    result is bit-for-bit what the dense path produced.
    """
    X = counts.astype(np.float64)
    rowsum = np.repeat(totals.astype(np.float64), np.diff(X.indptr))
    X.data = np.divide(X.data, rowsum, out=np.zeros_like(X.data), where=rowsum > 0) * target_sum
    np.log1p(X.data, out=X.data)
    return X


def sparse_moments(X):
    """Per-column mean and population standard deviation of a sparse matrix."""
    n = X.shape[0]
    data = X.data.astype(np.float64)
    total = np.bincount(X.indices, weights=data, minlength=X.shape[1])
    total_sq = np.bincount(X.indices, weights=data * data, minlength=X.shape[1])
    mean = total / n
    return mean, np.sqrt(np.maximum(total_sq / n - mean ** 2, 0.0))


def sparsepixels_to_pixelmatrix(values, source_genes, rows, x, y, markers):
    """
    Scatter a sparse ``(pixel, gene)`` matrix into a dense ``(y, x, marker)`` DataArray.

    ``values`` holds one row per surviving pixel and ``rows`` says where each
    belongs in the x-major grid spanned by ``x`` and ``y``; every unlisted grid
    position is left zero. Columns are pulled from ``source_genes`` in
    ``markers`` order, and markers absent from ``source_genes`` come out as zeros.
    """
    markers = pd.Index(markers)
    cols = source_genes.get_indexer(markers)
    present = np.flatnonzero(cols >= 0)

    dense = np.zeros((len(x) * len(y), len(markers)), np.float32)
    dense[np.ix_(rows, present)] = values[:, cols[present]].toarray()

    return xr.DataArray(
        dense.reshape(len(x), len(y), len(markers)).transpose(1, 0, 2),
        coords={'x': x, 'y': y, 'marker': markers.values},
        dims=['y', 'x', 'marker'],
    )


def sparsepixels_to_mask(rows, x, y):
    """Boolean ``(y, x)`` tissue mask marking the given rows of the pixel grid."""
    flat = np.zeros(len(x) * len(y), bool)
    flat[rows] = True
    return xr.DataArray(
        flat.reshape(len(x), len(y)).T,
        # the scalar 'marker' coord is a leftover of the old pivot-and-squeeze
        # construction; preserved so masks written now match those written before.
        coords={'x': x, 'y': y, 'marker': 'nonempty'},
        dims=['y', 'x'],
    )


def downsample(sample, factor, aggregate=np.mean):
    """Downsample a ``(y, x, marker)`` array by aggregating over ``factor``-by-``factor`` pixel blocks."""
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
    """Downsample a hi-res ``(y, x, marker)`` array by ``factor`` and wrap it as a named DataArray with micron coordinates."""
    sample = downsample(sample, factor)
    sample = xr.DataArray(
            sample,
            coords={'x': np.arange(sample.shape[1])*factor*pixelsize, 'y': np.arange(sample.shape[0])*factor*pixelsize, 'marker': markers},
            dims=['y', 'x', 'marker']
        ).astype(np.float32)
    sample.name = name
    return sample