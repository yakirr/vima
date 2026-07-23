# vima tests

## RA regression test (`test_ra_regression.py`)

An end-to-end **golden-master regression test**: it runs the full RA case-control
association (`vima.association`) and asserts the result — global p-value,
per-microniche `mncoef`, local `mncoef_fdr`, and the number of discoveries at
FDR ≤ 0.1 — hasn't drifted from a stored reference. Run it whenever you change
vima to catch unintended changes to the statistics.

### Running

```bash
pip install pytest        # one-time
pytest tests/             # or: python -m pytest tests/
```

- **First run** generates the golden reference and skips the comparisons. The
  reference is written to `tests/_testingdata/RA/association_reference.pkl`
  (a gitignored directory, so patient-derived numbers are never committed).
  Re-run to actually compare against it.
- **After an intentional change** to the statistics, refresh the baseline:

  ```bash
  VIMA_UPDATE_GOLDEN=1 pytest tests/
  ```

### Notes

- The RA data is **machine-local** (not in the repo). On a machine without it,
  the whole suite skips cleanly.
- The suite is **self-contained**: all inputs plus the generated golden
  reference live in the gitignored `tests/_testingdata/RA/` directory:
  - `fingerprints.h5ad`
  - `samples/*.nc`
  - `if-metadata.csv`
  - `association_reference.pkl` (generated)
- Comparison tolerances live at the top of `conftest.py` — tune them there.
- Because current vima may downsample microniches differently, comparisons are
  done on the microniches shared between the current run and the reference.

### Adding a portable test later

This suite depends on external RA data, so it can't run in CI or on a fresh
checkout. To get a CI-able test, add one built on a small synthetic (or
committed, downsampled) `Fingerprints` fixture and apply the same golden-master
pattern.
