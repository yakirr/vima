"""Characterization tests for the sparse rasterization path in vima.pp.st.

`st.get_sumstats` and `st.transcriptlist_to_normedpixelmatrix` used to build a
dense pixels-by-genes pandas table and pivot it. They now keep the counts in a
scipy CSR matrix throughout, which is ~14x faster and ~5x lighter on a Xenium 5K
panel. The rewrite is meant to be output-preserving, so the pre-rewrite
implementations are reproduced verbatim below and every test asserts the new
path still agrees with them.

The pixel matrices and masks must match *bit for bit*; the pooled moments are
allowed a tiny tolerance, because summing a column of a CSR with `bincount`
accumulates in a different order than `ndarray.mean(dtype=float64)` and the two
land up to ~1 float32 ULP apart.

Self-contained: the fixtures synthesize transcript tables, so these run anywhere.
"""
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import xarray as xr

import vima
from vima.ingest import st, util

PIXEL_SIZE = 10
TARGET_SUM = 50.0
MIN_NTX, MIN_NGENES = 11, 1
MOMENT_TOL = 1e-6   # generous next to the ~1e-8 relative drift actually seen


# --------------------------------------------------------------------------
# the pre-rewrite implementations, kept verbatim as the comparison baseline
# --------------------------------------------------------------------------
def ref_transcriptlist_to_pixellist(transcriptlist, x_col, y_col, gene_col, pixel_size=10):
    def complete(pl, colname, genes, fill=0.):
        vals = np.sort(pl[colname].unique())
        min_col = vals.min() // 1
        max_col = vals.max() // 1
        delta = int(min(vals[1:] - vals[:-1]))
        full_range = list(np.arange(min_col, max_col + 1, delta))
        locs_toadd = np.setdiff1d(full_range, vals)
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
    return complete(complete(pl, 'pixel_x', genes), 'pixel_y', genes)


def ref_df_to_xarray32(df):
    markers = df.columns.get_level_values('markers').unique()
    return xr.DataArray(
        df.values.reshape((len(df), len(markers), -1)).transpose(0, 2, 1),
        coords={'x': df.columns.get_level_values('pixel_x').unique().values,
                'y': df.index.values, 'marker': markers.values},
        dims=['y', 'x', 'marker']
    ).astype(np.float32)


def ref_pixellist_to_pixelmatrix(pl, markers):
    s = pd.pivot_table(pl, values=markers, index='pixel_y', columns='pixel_x').fillna(0)
    s.columns.names = ['markers', 'pixel_x']
    return ref_df_to_xarray32(s)


def ref_med_ntranscripts(df, x_col, y_col, pixel_size=PIXEL_SIZE):
    df = df.copy()
    df["px"] = (df[x_col] // pixel_size).astype(np.int32)
    df["py"] = (df[y_col] // pixel_size).astype(np.int32)
    pg = df.groupby(["px", "py"]).size().reset_index(name="txcount")
    return pg[pg.txcount > 10].txcount.median()


def ref_sample_sumstats(transcripts, x_col, y_col, gene_col, target_sum, n_top_genes,
                        min_mean=0.01, min_npixels=20, min_totalcounts=500):
    """One sample's contribution to get_sumstats: moments, npixels, and HVGs."""
    pl = ref_transcriptlist_to_pixellist(transcripts, x_col, y_col, gene_col, PIXEL_SIZE)
    genes = pl.columns[2:]
    obs = pl[['pixel_x', 'pixel_y']].copy()
    obs.index = obs.index.astype(str)
    ad = sc.AnnData(pl[genes].values.astype(np.float32), var=pd.DataFrame(index=genes), obs=obs)
    sc.pp.filter_cells(ad, min_counts=MIN_NTX, inplace=True)
    sc.pp.filter_cells(ad, min_genes=MIN_NGENES, inplace=True)

    X = np.log1p(ad.X / ad.X.sum(axis=1, keepdims=True) * target_sum)
    mean = pd.Series(np.array(X.mean(axis=0, dtype=np.float64)).squeeze(), index=genes)
    std = pd.Series(np.array(X.std(axis=0, dtype=np.float64)).squeeze(), index=genes)
    npixels = ad.n_obs

    sc.pp.filter_genes(ad, min_cells=min_npixels, inplace=True)
    sc.pp.filter_genes(ad, min_counts=min_totalcounts, inplace=True)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, flavor='seurat_v3', subset=False)
    hvgs = ad.var_names[ad.var.highly_variable & (ad.var.means >= min_mean)].tolist()
    return mean, std, npixels, hvgs, list(genes)


def ref_normedpixelmatrix(sid, data, x_col, y_col, gene_col, target_sum, means, stds, genes):
    pl = ref_transcriptlist_to_pixellist(data, x_col, y_col, gene_col, PIXEL_SIZE)
    markers = pl.columns[2:]
    all_x = np.sort(pl['pixel_x'].unique())
    all_y = np.sort(pl['pixel_y'].unique())

    pl = pl[(pl[markers] != 0).sum(axis=1) >= MIN_NGENES]
    pl = pl[(pl[markers].sum(axis=1) >= MIN_NTX)]
    pl[markers] = pl[markers].div(pl[markers].sum(axis=1), axis=0).fillna(0) * target_sum
    pl[markers] = np.log1p(pl[markers])

    pl = pl.reindex(columns=['pixel_x', 'pixel_y'] + genes, fill_value=0)
    mask_pl = pl[['pixel_x', 'pixel_y']].copy()
    mask_pl['nonempty'] = 1

    s = ref_pixellist_to_pixelmatrix(pl, genes).reindex({'x': all_x, 'y': all_y}, fill_value=0)
    s.attrs['means'] = means.reindex(s.marker.values).values.astype(np.float32)
    s.attrs['stds'] = stds.reindex(s.marker.values).values.astype(np.float32)
    mask = (ref_pixellist_to_pixelmatrix(mask_pl, ['nonempty']).squeeze().astype(bool)
            .reindex({'x': all_x, 'y': all_y}, fill_value=False))
    s.name = sid
    mask.name = sid
    return mask, s.astype(np.float32)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
ALL_GENES = [f"G{i:02d}" for i in range(40)]


def synth(seed, genes=ALL_GENES, n_tx=120_000, nx=40, ny=30, categorical=False,
          categories=None, nan_every=0):
    """A transcript table with spatially structured, gene-dependent density."""
    rng = np.random.default_rng(seed)
    # each gene sits in a band along x, so per-gene variance differs and the
    # HVG ranking is meaningful rather than noise
    centers = rng.uniform(0, nx * PIXEL_SIZE, len(genes))
    widths = rng.uniform(0.1, 0.9, len(genes)) * nx * PIXEL_SIZE
    counts = rng.multinomial(n_tx, rng.dirichlet(np.full(len(genes), 4.0)))

    xs, ys, gs = [], [], []
    for g, c, w, k in zip(genes, centers, widths, counts):
        xs.append(np.clip(rng.normal(c, w, k), 0, nx * PIXEL_SIZE - 1e-6))
        ys.append(rng.uniform(0, ny * PIXEL_SIZE, k))
        gs.append(np.full(k, g, dtype=object))
    x = np.concatenate(xs); y = np.concatenate(ys); g = np.concatenate(gs)
    order = rng.permutation(len(x))
    x, y, g = x[order], y[order], g[order]

    if nan_every:
        g = g.copy()
        g[::nan_every] = None
    if categorical:
        g = pd.Categorical(g, categories=categories if categories is not None else sorted(genes))
    return pd.DataFrame({"x_location": x, "y_location": y, "feature_name": g})


@pytest.fixture(scope="module")
def samples():
    """Three samples; the third is missing a chunk of the panel."""
    return {
        "s1": synth(0),
        "s2": synth(1, n_tx=90_000, nx=35, ny=45),
        "s3": synth(2, genes=ALL_GENES[:28], n_tx=100_000),   # 12 genes absent
    }


def assert_matrices_identical(got, exp):
    """The DataArrays must agree exactly, coords, dtypes, attrs and all."""
    assert got.dims == exp.dims
    assert got.shape == exp.shape
    assert got.name == exp.name
    assert got.values.dtype == exp.values.dtype
    for c in ("x", "y", "marker"):
        assert np.array_equal(got[c].values, exp[c].values), f"{c} coord differs"
        assert got[c].values.dtype == exp[c].values.dtype, f"{c} coord dtype differs"
    assert np.array_equal(got.values, exp.values), "pixel values are not bitwise identical"
    for k in ("means", "stds"):
        np.testing.assert_allclose(got.attrs[k], exp.attrs[k], rtol=MOMENT_TOL, atol=1e-7)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_med_ntranscripts_matches_dense(samples, tmp_path):
    paths = list(samples)
    load = lambda sid: (sid, samples[sid])
    got = st.med_ntranscripts(load, paths, "x_location", "y_location", pixel_size=PIXEL_SIZE)
    exp = np.mean([ref_med_ntranscripts(df, "x_location", "y_location") for df in samples.values()])
    assert got == exp


def test_sumstats_matches_dense(samples):
    """HVG selection is exact; pooled moments match to float32 rounding."""
    pytest.importorskip("skmisc", reason="seurat_v3 HVG selection needs scikit-misc")
    paths = list(samples)
    load = lambda sid: (sid, samples[sid])
    n_top = 15

    hvgs, mean, std = st.get_sumstats(
        load, paths, TARGET_SUM, "x_location", "y_location", "feature_name",
        n_top_genes_per_sample=n_top, pixel_size=PIXEL_SIZE,
        min_ntranscripts_per_pixel=MIN_NTX, min_ngenes_per_pixel=MIN_NGENES)

    means, stds, npix, exp_hvgs, allgenes = [], [], [], set(), set()
    for df in samples.values():
        m, s, n, h, g = ref_sample_sumstats(df, "x_location", "y_location", "feature_name",
                                            TARGET_SUM, n_top)
        means.append(m); stds.append(s); npix.append(n)
        exp_hvgs.update(h); allgenes.update(g)
    exp_mean, exp_std = util.pool_moments(
        pd.concat([m.reindex(index=allgenes, fill_value=0) for m in means], axis=1),
        pd.concat([s.reindex(index=allgenes, fill_value=0) for s in stds], axis=1),
        npix)

    assert set(hvgs) == exp_hvgs
    assert set(mean.index) == set(exp_mean.index)
    np.testing.assert_allclose(mean.values, exp_mean.reindex(mean.index).values, rtol=MOMENT_TOL, atol=1e-9)
    np.testing.assert_allclose(std.values, exp_std.reindex(std.index).values, rtol=MOMENT_TOL, atol=1e-9)


@pytest.mark.parametrize("sid", ["s1", "s2", "s3"])
def test_pixelmatrix_is_bitwise_identical(samples, sid):
    """The headline guarantee: same pixel matrix, same mask, to the last bit."""
    genes = list(ALL_GENES)     # includes genes absent from s3
    means = pd.Series(np.linspace(0.1, 2.0, len(genes)), index=genes)
    stds = pd.Series(np.linspace(0.5, 1.5, len(genes)), index=genes)

    mask, pm = st.transcriptlist_to_normedpixelmatrix(
        sid, samples[sid], "x_location", "y_location", "feature_name", PIXEL_SIZE,
        TARGET_SUM, means=means, stds=stds, genes=genes,
        min_ngenes_per_pixel=MIN_NGENES, min_ntranscripts_per_pixel=MIN_NTX)
    exp_mask, exp_pm = ref_normedpixelmatrix(
        sid, samples[sid], "x_location", "y_location", "feature_name",
        TARGET_SUM, means, stds, genes)

    assert_matrices_identical(pm, exp_pm)
    assert mask.dims == exp_mask.dims
    assert mask.name == exp_mask.name
    assert np.array_equal(mask.values, exp_mask.values)
    assert np.array_equal(mask.x.values, exp_mask.x.values)
    assert np.array_equal(mask.y.values, exp_mask.y.values)
    # the mask carries a scalar 'marker' coord left over from the old
    # pivot-and-squeeze construction; preserved so old and new .nc files match
    assert mask.coords["marker"].values == exp_mask.coords["marker"].values


def test_genes_absent_from_a_sample_become_zero_slices(samples):
    """s3 lacks 12 of the 40 requested genes; those slices must be all zero."""
    genes = list(ALL_GENES)
    absent = ALL_GENES[28:]
    assert not set(absent) & set(samples["s3"].feature_name.unique())
    moments = pd.Series(1.0, index=genes)

    _, pm = st.transcriptlist_to_normedpixelmatrix(
        "s3", samples["s3"], "x_location", "y_location", "feature_name", PIXEL_SIZE,
        TARGET_SUM, means=moments, stds=moments, genes=genes,
        min_ngenes_per_pixel=MIN_NGENES, min_ntranscripts_per_pixel=MIN_NTX)

    assert list(pm.marker.values) == sorted(genes)
    assert pm.sel(marker=absent).values.max() == 0
    assert pm.sel(marker=ALL_GENES[:28]).values.max() > 0


def test_unrequested_genes_are_dropped(samples):
    """Asking for a subset yields exactly that subset, sorted."""
    genes = ["G05", "G00", "G31"]
    moments = pd.Series(1.0, index=ALL_GENES)
    _, pm = st.transcriptlist_to_normedpixelmatrix(
        "s1", samples["s1"], "x_location", "y_location", "feature_name", PIXEL_SIZE,
        TARGET_SUM, means=moments, stds=moments, genes=genes,
        min_ngenes_per_pixel=MIN_NGENES, min_ntranscripts_per_pixel=MIN_NTX)
    assert list(pm.marker.values) == ["G00", "G05", "G31"]


def test_categorical_gene_column_keeps_unobserved_levels():
    """A declared-but-unobserved category stays as an all-zero column, as before."""
    df = synth(3, categorical=True, categories=ALL_GENES + ["NEVER_SEEN"])
    pixels = util.transcriptlist_to_sparsepixels(df, "x_location", "y_location", "feature_name",
                                                 pixel_size=PIXEL_SIZE)
    exp = ref_transcriptlist_to_pixellist(df, "x_location", "y_location", "feature_name", PIXEL_SIZE)
    assert list(pixels.genes) == list(exp.columns[2:])
    assert "NEVER_SEEN" in pixels.genes
    assert pixels.counts[:, pixels.genes.get_loc("NEVER_SEEN")].nnz == 0


def test_transcripts_with_missing_gene_are_dropped():
    """NaN genes are skipped, matching what value_counts used to do."""
    df = synth(4, nan_every=7)
    pixels = util.transcriptlist_to_sparsepixels(df, "x_location", "y_location", "feature_name",
                                                 pixel_size=PIXEL_SIZE)
    exp = ref_transcriptlist_to_pixellist(df, "x_location", "y_location", "feature_name", PIXEL_SIZE)
    assert list(pixels.genes) == list(exp.columns[2:])
    assert pixels.counts.sum() == exp[exp.columns[2:]].values.sum()
    assert pixels.counts.sum() == df.feature_name.notna().sum()


def test_sparse_counts_match_the_dense_pixel_table(samples):
    """The CSR holds exactly the counts the dense pixel table used to hold."""
    for sid, df in samples.items():
        pixels = util.transcriptlist_to_sparsepixels(df, "x_location", "y_location", "feature_name",
                                                     pixel_size=PIXEL_SIZE)
        exp = ref_transcriptlist_to_pixellist(df, "x_location", "y_location", "feature_name", PIXEL_SIZE)
        genes = list(exp.columns[2:])
        assert list(pixels.genes) == genes, sid

        # line the dense table's rows up with the CSR's grid rows
        rows = ((exp.pixel_x.to_numpy() // PIXEL_SIZE - pixels.x[0] // PIXEL_SIZE) * len(pixels.y)
                + (exp.pixel_y.to_numpy() // PIXEL_SIZE - pixels.y[0] // PIXEL_SIZE))
        got = np.asarray(pixels.counts[rows].todense())
        assert np.array_equal(got, exp[genes].to_numpy().astype(np.float32)), sid
