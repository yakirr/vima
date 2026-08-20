"""Tests for vima.pp.collapse_markers and the moment-pooling it shares with ingest.

Unlike the RA regression suite, these are self-contained: the fixtures synthesize
a tiny normalized dataset on the fly, so they run anywhere.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import vima

MARKERS = ["CD3E", "CD19", "COL1A1", "ACTA2", "KRT5"]
KEEP = ["CD3E", "CD19"]
DROP = [m for m in MARKERS if m not in KEEP]
SHAPES = {"s1": (6, 5), "s2": (4, 7)}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A two-sample normalized/masks pair, as prepare_* would have written it."""
    root = tmp_path_factory.mktemp("collapse")
    normeddir, masksdir = root / "normalized", root / "masks"
    normeddir.mkdir()
    masksdir.mkdir()

    rng = np.random.default_rng(0)
    means = pd.Series(rng.uniform(0.1, 2.0, len(MARKERS)), index=MARKERS)
    stds = pd.Series(rng.uniform(0.5, 1.5, len(MARKERS)), index=MARKERS)

    src, masks = {}, {}
    for sid, (ny, nx) in SHAPES.items():
        mask = rng.random((ny, nx)) > 0.3
        data = np.abs(rng.normal(0, 1, (ny, nx, len(MARKERS)))).astype(np.float32)
        data[~mask] = 0  # empty pixels are zero, as both prepare paths guarantee
        s = xr.DataArray(
            data,
            dims=["y", "x", "marker"],
            coords={"x": np.arange(nx) * 10.0, "y": np.arange(ny) * 10.0, "marker": MARKERS},
        )
        s.name = sid
        s.attrs["means"] = means.values.astype(np.float32)
        s.attrs["stds"] = stds.values.astype(np.float32)
        m = xr.DataArray(mask, dims=["y", "x"], coords={"x": s.x, "y": s.y})
        m.name = sid
        vima.pp.util.write_xarray(s, f"{normeddir}/{sid}.nc")
        vima.pp.util.write_xarray(m, f"{masksdir}/{sid}.nc")
        src[sid], masks[sid] = s, mask

    return {"root": root, "normeddir": str(normeddir), "masksdir": str(masksdir),
            "src": src, "masks": masks, "means": means, "stds": stds}


@pytest.fixture(scope="module")
def collapsed(dataset, tmp_path_factory):
    """Run collapse_markers once, including a marker that isn't in the data."""
    outdir = str(tmp_path_factory.mktemp("collapsed"))
    vima.pp.collapse_markers(dataset["normeddir"], KEEP + ["NOT_A_GENE"], "other", outdir)
    return outdir


@pytest.mark.parametrize("sid", list(SHAPES))
def test_marker_set(collapsed, sid):
    with xr.open_dataarray(f"{collapsed}/{sid}.nc") as s:
        assert list(s.marker.values) == KEEP + ["other"]


@pytest.mark.parametrize("sid", list(SHAPES))
def test_retained_values_unchanged(dataset, collapsed, sid):
    with xr.open_dataarray(f"{collapsed}/{sid}.nc") as s:
        for m in KEEP:
            assert np.array_equal(s.sel(marker=m).values, dataset["src"][sid].sel(marker=m).values)


@pytest.mark.parametrize("sid", list(SHAPES))
def test_pseudomarker_is_log_of_summed_counts(dataset, collapsed, sid):
    """The collapsed channel is log1p of the summed normalized counts of the rest."""
    expected = np.log1p(
        np.expm1(dataset["src"][sid].sel(marker=DROP).values.astype(np.float64)).sum(-1))
    with xr.open_dataarray(f"{collapsed}/{sid}.nc") as s:
        assert np.allclose(s.sel(marker="other").values, expected, atol=1e-5)


@pytest.mark.parametrize("sid", list(SHAPES))
def test_empty_pixels_stay_empty(dataset, collapsed, sid):
    with xr.open_dataarray(f"{collapsed}/{sid}.nc") as s:
        assert (s.values[~dataset["masks"][sid]] == 0).all()


@pytest.mark.parametrize("sid", list(SHAPES))
def test_retained_moments_carried_over(dataset, collapsed, sid):
    with xr.open_dataarray(f"{collapsed}/{sid}.nc") as s:
        assert len(s.attrs["means"]) == len(s.attrs["stds"]) == len(KEEP) + 1
        assert np.array_equal(s.attrs["means"][:-1], dataset["means"][KEEP].values.astype(np.float32))
        assert np.array_equal(s.attrs["stds"][:-1], dataset["stds"][KEEP].values.astype(np.float32))


def test_pseudomarker_moments_are_pooled_over_masked_pixels(dataset, collapsed):
    """The new moments match pool_moments over each sample's non-empty pixels."""
    sids = list(SHAPES)
    per_sample = {}
    for sid in sids:
        with xr.open_dataarray(f"{collapsed}/{sid}.nc") as s:
            vals = s.sel(marker="other").values[dataset["masks"][sid]]
        per_sample[sid] = (vals.mean(dtype=np.float64), vals.std(dtype=np.float64), len(vals))

    exp_mean, exp_std = vima.pp.util.pool_moments(
        pd.DataFrame([[per_sample[s][0] for s in sids]], index=["other"], columns=sids),
        pd.DataFrame([[per_sample[s][1] for s in sids]], index=["other"], columns=sids),
        [per_sample[s][2] for s in sids])

    for sid in sids:  # every sample carries the same dataset-wide moments
        with xr.open_dataarray(f"{collapsed}/{sid}.nc") as s:
            assert np.isclose(s.attrs["means"][-1], exp_mean.iloc[0], atol=1e-5)
            assert np.isclose(s.attrs["stds"][-1], exp_std.iloc[0], atol=1e-5)


def test_name_collision_raises(dataset, tmp_path):
    with pytest.raises(ValueError, match="already a marker"):
        vima.pp.collapse_markers(dataset["normeddir"], KEEP, "CD3E", str(tmp_path / "bad"))


def test_no_retained_markers_raises(dataset, tmp_path):
    with pytest.raises(ValueError, match="None of the requested markers"):
        vima.pp.collapse_markers(dataset["normeddir"], ["NOPE"], "other", str(tmp_path / "bad"))


def test_output_feeds_the_standardization_path(dataset, collapsed):
    """dimreduce applies the attrs positionally, so this catches misalignment."""
    mps, _ = vima.pp.dimreduce.metapixels_allsamples(
        collapsed, dataset["masksdir"], list(SHAPES), total_n_metapixels=20)
    for mp in mps.values():
        assert list(mp.columns) == KEEP + ["other"]
        assert np.isfinite(mp.values).all()


def test_pool_moments_matches_inlined_formula():
    """pool_moments was extracted from st.get_sumstats; it must not have drifted."""
    rng = np.random.default_rng(1)
    idx = [f"g{i}" for i in range(7)]
    means_df = pd.DataFrame(rng.normal(1, 0.5, (7, 4)), index=idx)
    stds_df = pd.DataFrame(rng.uniform(0.2, 2, (7, 4)), index=idx)
    npixels = [1000, 250, 7, 4321]

    w = np.array(npixels, dtype=np.float64)
    W = w.sum()
    grand_mean = np.sum((means_df * w).values, axis=1, dtype=np.float64) / W
    mean_of_vars = np.sum((stds_df**2 * w).values, axis=1, dtype=np.float64) / W
    var_of_means = (
        (means_df.subtract(grand_mean, axis=0).values.astype(np.float64) ** 2) * w
    ).sum(axis=1) / W

    got_mean, got_std = vima.pp.util.pool_moments(means_df, stds_df, npixels)
    assert got_mean.equals(pd.Series(grand_mean, index=idx))
    assert got_std.equals(pd.Series(np.sqrt(mean_of_vars + var_of_means), index=idx))
