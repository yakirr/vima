import time
import cProfile
import pstats
import io
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
import vima
from vima.composition import cell_type_counts

# ── synthetic test data ────────────────────────────────────────────────────────
np.random.seed(42)
N_SAMPLES = 5
N_PATCHES = 10_000
N_CELLS   = 10_000
PATCHSIZE = 40                  # pixels
PIXEL_UM  = 10                  # microns per pixel → patch spans 400 um
IMG_UM    = 8_000               # image extent in microns

CELL_TYPES = ['T', 'B', 'NK', 'Myeloid', 'Epithelial']
sids = [f'sample_{i}' for i in range(N_SAMPLES)]

patch_sids = np.tile(sids, N_PATCHES // N_SAMPLES)
patch_meta = pd.DataFrame({
    'sid':       patch_sids,
    'x_microns': np.random.uniform(0, IMG_UM - PATCHSIZE * PIXEL_UM, N_PATCHES),
    'y_microns': np.random.uniform(0, IMG_UM - PATCHSIZE * PIXEL_UM, N_PATCHES),
    'patchsize': PATCHSIZE,
})

cell_sids = np.tile(sids, N_CELLS // N_SAMPLES)
cells = pd.DataFrame({
    'sid':      cell_sids,
    'x':        np.random.uniform(0, IMG_UM, N_CELLS),
    'y':        np.random.uniform(0, IMG_UM, N_CELLS),
    'celltype': np.random.choice(CELL_TYPES, N_CELLS),
})

print(f'Patches: {N_PATCHES}  |  Cells: {N_CELLS}  |  Samples: {N_SAMPLES}')
print(f'Patch size: {PATCHSIZE} px = {PATCHSIZE * PIXEL_UM} um\n')


# ── reference implementation ───────────────────────────────────────────────────
def add_cellcounts(obs, celltype_col):
    obs = obs.copy()
    for sid in obs.sid.unique():
        mycells   = cells[cells.sid == sid]
        mypatches = obs[obs.sid == sid]

        cellcounts = pd.concat([
            mycells[
                (mycells.x >= x) & (mycells.x <= x + PIXEL_UM * ps) &
                (mycells.y >= y) & (mycells.y <= y + PIXEL_UM * ps)
            ][celltype_col].value_counts()
            for _, x, y, ps in mypatches[['sid', 'x_microns', 'y_microns', 'patchsize']].values
        ], axis=1).T.fillna(0)

        idx = mypatches.index
        for c in cellcounts.columns:
            obs.loc[idx, c] = cellcounts[c].to_numpy()
    return obs


# ── wall-clock timing ──────────────────────────────────────────────────────────
print('=== Wall-clock timing ===')

t0 = time.perf_counter()
result_new = cell_type_counts(
    cells, patch_meta,
    sid_col='sid', celltype_col='celltype', x_col='x', y_col='y',
    patch_x_col='x_microns', patch_y_col='y_microns',
    patch_size_col='patchsize', pixel_size_microns=PIXEL_UM,
)
t_new = time.perf_counter() - t0
print(f'  cell_type_counts (new):  {t_new:.3f}s')

t0 = time.perf_counter()
result_ref = add_cellcounts(patch_meta, 'celltype')
t_ref = time.perf_counter() - t0
print(f'  add_cellcounts (ref):    {t_ref:.3f}s')
print(f'  speedup: {t_ref / t_new:.1f}x\n')


# ── correctness spot-check ─────────────────────────────────────────────────────
# Both should agree on total cell assignments (sum across all patches and types)
new_total = result_new.values.sum()
ref_cols  = [c for c in result_ref.columns if c in CELL_TYPES]
ref_total = result_ref[ref_cols].values.sum()
print(f'=== Correctness check ===')
print(f'  new total cell assignments: {int(new_total)}')
print(f'  ref total cell assignments: {int(ref_total)}')
print(f'  match: {new_total == ref_total}\n')


# ── cProfile breakdown ─────────────────────────────────────────────────────────
def run_new():
    cell_type_counts(
        cells, patch_meta,
        sid_col='sid', celltype_col='celltype', x_col='x', y_col='y',
        patch_x_col='x_microns', patch_y_col='y_microns',
        patch_size_col='patchsize', pixel_size_microns=PIXEL_UM,
    )

def run_ref():
    add_cellcounts(patch_meta, 'celltype')


for label, fn in [('cell_type_counts (new)', run_new),
                   ('add_cellcounts (ref)',   run_ref)]:
    print(f'=== cProfile: {label} ===')
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats('cumulative')
    ps.print_stats(15)
    print(buf.getvalue())
