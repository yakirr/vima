"""End-to-end regression test: the RA case-control association result must stay
stable across changes to vima. Compares the current pipeline output against the
stored golden master (see conftest.py) within tolerance.

Run:              pytest tests/
Update baseline:  VIMA_UPDATE_GOLDEN=1 pytest tests/     # after an intentional change
"""
import numpy as np

from conftest import (
    MNCOEF_CORR_MIN,
    MNCOEF_MAXDIFF,
    FDR_CORR_MIN,
    SHARED_FRAC_MIN,
    FDR_COUNT_REL_TOL,
    FDR_COUNT_ABS_TOL,
)


def _nlog10(fdr):
    return -np.log10(np.clip(np.asarray(fdr, dtype=float), 1e-300, None))


def _aligned(ra_result, reference):
    """Restrict both results to the microniches present in both (current vima may
    downsample differently), preserving reference order."""
    cur, ref = ra_result["obs"], reference["obs"]
    idx = ref.index.intersection(cur.index)
    return cur.loc[idx], ref.loc[idx]


def test_shared_microniches(ra_result, reference):
    _, ref = _aligned(ra_result, reference)
    frac = len(ref) / len(reference["obs"])
    assert frac >= SHARED_FRAC_MIN, (
        f"only {frac:.1%} of reference microniches present in current run "
        f"({len(ref)}/{len(reference['obs'])})"
    )


def test_global_p(ra_result, reference):
    cur_p, ref_p = ra_result["p"], reference["p"]
    np.testing.assert_allclose(
        cur_p, ref_p, atol=1e-6, rtol=1e-3,
        err_msg=f"global p changed from {ref_p:.6g} to {cur_p:.6g}",
    )


def test_mncoef_regression(ra_result, reference):
    cur, ref = _aligned(ra_result, reference)
    r = np.corrcoef(cur.mncoef.values, ref.mncoef.values)[0, 1]
    maxdiff = np.abs(cur.mncoef.values - ref.mncoef.values).max()
    assert r >= MNCOEF_CORR_MIN, f"mncoef correlation dropped to {r:.5f}"
    assert maxdiff <= MNCOEF_MAXDIFF, f"max |Δmncoef| = {maxdiff:.2e}"


def test_fdr_regression(ra_result, reference):
    cur, ref = _aligned(ra_result, reference)
    r = np.corrcoef(_nlog10(cur.mncoef_fdr), _nlog10(ref.mncoef_fdr))[0, 1]
    assert r >= FDR_CORR_MIN, f"-log10(fdr) correlation dropped to {r:.5f}"


def test_discovery_count(ra_result, reference):
    cur_n = int((ra_result["obs"].mncoef_fdr <= 0.1).sum())
    ref_n = int((reference["obs"].mncoef_fdr <= 0.1).sum())
    tol = max(FDR_COUNT_ABS_TOL, int(FDR_COUNT_REL_TOL * ref_n))
    assert abs(cur_n - ref_n) <= tol, (
        f"#(mncoef_fdr <= 0.1) changed from {ref_n} to {cur_n} (tol ±{tol})"
    )
