"""Shared fixtures and configuration for vima's regression tests.

The RA case-control association is used as an end-to-end regression ("golden
master") check: we run the full pipeline and compare its output to a stored
reference. The reference is generated from the current code the first time the
suite runs (or when VIMA_UPDATE_GOLDEN=1), and committed *nowhere* -- it lives
next to the (gitignored) RA data so patient-derived numbers stay out of the repo.

Because that data is machine-local (not in the repo, not in CI), the whole suite
skips cleanly when the data is absent.
"""
import os
import glob
import pickle
from pathlib import Path

import pytest

# --- locations -------------------------------------------------------------
# Everything the suite needs is self-contained in the gitignored _testingdata dir.
RA_DATA = Path(__file__).resolve().parent / "_testingdata" / "RA"
FINGERPRINTS = RA_DATA / "fingerprints.h5ad"
SAMPLES_GLOB = str(RA_DATA / "samples" / "*.nc")
METADATA_CSV = RA_DATA / "if-metadata.csv"
REFERENCE = RA_DATA / "association_reference.pkl"  # generated golden master

UPDATE_GOLDEN = os.environ.get("VIMA_UPDATE_GOLDEN") == "1"

# --- comparison tolerances (tune to taste) ---------------------------------
# On the same machine + unchanged code these are effectively exact; the slack is
# to ignore pure floating-point reordering while still catching real drift.
MNCOEF_CORR_MIN = 0.9999      # Pearson r between old/new per-microniche mncoef
MNCOEF_MAXDIFF = 1e-4         # max |Δmncoef| across shared microniches
FDR_CORR_MIN = 0.999          # Pearson r between old/new -log10(mncoef_fdr)
SHARED_FRAC_MIN = 0.95        # fraction of reference microniches still present
FDR_COUNT_REL_TOL = 0.02      # allowed drift in #(fdr <= 0.1), relative
FDR_COUNT_ABS_TOL = 5         # ...or this many, whichever is larger


def _data_available():
    return (
        FINGERPRINTS.exists()
        and METADATA_CSV.exists()
        and len(glob.glob(SAMPLES_GLOB)) > 0
    )


@pytest.fixture(scope="session")
def ra_result():
    """Run the RA case-control association with the CURRENT vima and return
    {p, obs}. Session-scoped so the (~15s) pipeline runs once for all tests."""
    if not _data_available():
        pytest.skip("RA data not available on this machine; skipping regression suite")

    import torch
    import pandas as pd
    import vima

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_default_device(device)

    cwd = os.getcwd()
    os.chdir(RA_DATA)  # contain any incidental outputs within the gitignored dir
    try:
        samples = vima.read_samples(SAMPLES_GLOB)

        fullmeta = pd.read_csv(METADATA_CSV).set_index("subject_id")[["CTAP"]]
        fullmeta.index = fullmeta.index.str.replace("V0", "")
        fullmeta["fstar"] = fullmeta.CTAP.isin(["F", "T + F", "E + F + M"])

        sid_to_donor = {
            s.sid: s.sid.split("_")[0].replace("Repeat", "") for s in samples.values()
        }
        samplemeta = vima.reindex_by_sid(fullmeta, sid_to_donor)

        ds = vima.Fingerprints.read_h5ad(str(FINGERPRINTS))
        ds = vima.Fingerprints.from_list([ds.select_model(i) for i in range(10)])

        p, D = vima.association(
            ds, samplemeta.fstar, "sid", donorids=samplemeta.donor, make_umap=False
        )
    finally:
        os.chdir(cwd)

    return {"p": float(p), "obs": D.obs[["mncoef", "mncoef_fdr"]].copy(), "device": device}


@pytest.fixture(scope="session")
def reference(ra_result):
    """The stored golden master. On first run (or VIMA_UPDATE_GOLDEN=1) it is
    written from the current result and the comparison tests skip -- rerun to
    actually compare against it."""
    if UPDATE_GOLDEN or not REFERENCE.exists():
        with open(REFERENCE, "wb") as f:
            pickle.dump(ra_result, f)
        pytest.skip(f"golden reference written to {REFERENCE}; re-run to compare against it")
    with open(REFERENCE, "rb") as f:
        return pickle.load(f)
