import os
import shutil
import tarfile
import tempfile

import cv2
import numpy as np
import pandas as pd
import scanpy as sc
import xarray as xr
import vima

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
RESOLUTION_UM = 10
PATCH_SIZE_UM = 400
N_SECRET = 20
SECRET_MEAN_FRAC = 0.05
DROPOUT_PROB = 0.10

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, '../../../../ST/ALZ/alz-data/transcripts')
CELLS_FILE = os.path.join(HERE, '../../../../ST/ALZ/alz-data/SEAAD_MTG_MERFISH_metadata.2024-05-03.noblanks.harmonized.txt')
H5AD_FILE = os.path.join(HERE, '../../../../ST/vimapaper/ALZ/_results/cc_dementia.h5ad')
OUT_DIR = os.path.join(HERE, 'ST/raw')


def get_l23it_region(df, sid_cells, resolution=RESOLUTION_UM):
    """Rasterize L2/3 IT cells and return a boolean mask of the dense region."""
    x_min, x_max = df['global_x'].min(), df['global_x'].max()
    y_min, y_max = df['global_y'].min(), df['global_y'].max()

    xs = np.arange(x_min, x_max + resolution, resolution)
    ys = np.arange(y_min, y_max + resolution, resolution)
    layer = xr.DataArray(
        np.zeros((len(ys), len(xs)), dtype=np.uint8),
        dims=['y', 'x'],
        coords={'y': ys, 'x': xs},
    )

    mycells_ = sid_cells[sid_cells.subclass_name == 'L2/3 IT']
    for cx, cy in mycells_[['x', 'y']].values:
        nearest = layer.sel(x=cx, y=cy, method='nearest')
        layer.loc[nearest.y.item(), nearest.x.item()] = 1

    layer.data = cv2.morphologyEx(
        layer.data, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    )
    layer.data = cv2.morphologyEx(
        layer.data, cv2.MORPH_OPEN,
        np.ones((20, 20), np.uint8)
    )
    return layer.astype(bool)


def transcripts_in_region(df, region, xcol='x', ycol='y'):
    """Boolean array True where each row's (xcol, ycol) falls inside the region mask."""
    x0 = region.x.values[0]
    y0 = region.y.values[0]
    dx = float(region.x.values[1] - region.x.values[0])
    dy = float(region.y.values[1] - region.y.values[0])
    nx, ny = len(region.x), len(region.y)

    xi = np.round((df[xcol].values - x0) / dx).astype(int)
    yi = np.round((df[ycol].values - y0) / dy).astype(int)

    valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    in_region = np.zeros(len(df), dtype=bool)
    in_region[valid] = region.values[yi[valid], xi[valid]]
    return in_region


def l23_region_size(sid_cells):
    """Count L2/3 IT mask pixels using the cells bounding box as raster extent."""
    if sid_cells.empty or (sid_cells.subclass_name == 'L2/3 IT').sum() == 0:
        return 0
    proxy_df = pd.DataFrame({
        'global_x': [sid_cells['x'].min(), sid_cells['x'].max()],
        'global_y': [sid_cells['y'].min(), sid_cells['y'].max()],
    })
    return int(get_l23it_region(proxy_df, sid_cells).values.sum())


def find_patch_corner(mask):
    """Return (yi, xi) pixel indices of a random 400×400µm window fully inside mask."""
    pw = PATCH_SIZE_UM // RESOLUTION_UM
    m = mask.values.astype(np.int32)
    padded = np.pad(m.cumsum(0).cumsum(1), [[1, 0], [1, 0]])
    window_sum = padded[pw:, pw:] - padded[:-pw, pw:] - padded[pw:, :-pw] + padded[:-pw, :-pw]
    valid_yx = np.argwhere(window_sum == pw * pw)
    if len(valid_yx) == 0:
        raise ValueError('No fully-contained 400×400µm patch found in L2/3 region')
    return valid_yx[rng.integers(len(valid_yx))]


def tile_l23(df, region, patch_df, patch_origin, secret_genes_orig, n_patch):
    """Replace L2/3 transcripts with randomly-placed patch copies plus added SECRET signal.

    Three phases:
      1. Sample random positions until 99% of the L2/3 region is covered (numpy only).
      2. Build a pixel-level ownership map (last-write wins) via cheap array slices.
      3. Generate transcripts per tile; keep only those whose pixel is owned by that tile.
    """
    px0, py0 = patch_origin
    l23_x = region.x.values
    l23_y = region.y.values
    l23_x0, l23_dx = float(l23_x[0]), float(l23_x[1] - l23_x[0])
    l23_y0, l23_dy = float(l23_y[0]), float(l23_y[1] - l23_y[0])
    nx, ny = len(l23_x), len(l23_y)
    total_l23_pixels = int(region.values.sum())

    if total_l23_pixels == 0:
        return df

    # Phase 1: sample positions until 99% of L2/3 pixels are covered
    print(f'  Phase 1: sampling patch positions (L2/3 region: {total_l23_pixels} pixels)...')
    coverage = np.zeros((ny, nx), dtype=bool)
    positions, slices = [], []
    while coverage.sum() < 0.99 * total_l23_pixels and len(positions) < 5000:
        rx0 = rng.uniform(l23_x[0] - PATCH_SIZE_UM, l23_x[-1])
        ry0 = rng.uniform(l23_y[0] - PATCH_SIZE_UM, l23_y[-1])
        xi_lo = max(0, int(np.searchsorted(l23_x, rx0)))
        xi_hi = min(nx, int(np.searchsorted(l23_x, rx0 + PATCH_SIZE_UM, side='right')))
        yi_lo = max(0, int(np.searchsorted(l23_y, ry0)))
        yi_hi = min(ny, int(np.searchsorted(l23_y, ry0 + PATCH_SIZE_UM, side='right')))
        if xi_hi > xi_lo and yi_hi > yi_lo:
            coverage[yi_lo:yi_hi, xi_lo:xi_hi] |= region.values[yi_lo:yi_hi, xi_lo:xi_hi]
        positions.append((rx0, ry0))
        slices.append((xi_lo, xi_hi, yi_lo, yi_hi))
        if len(positions) % 50 == 0:
            print(f'    patch {len(positions)}: coverage {coverage.sum() / total_l23_pixels:.1%}', flush=True)

    print(f'  Phase 1 done: {len(positions)} patches, coverage {coverage.sum() / total_l23_pixels:.1%}')

    # Phase 2: build ownership map — each L2/3 pixel belongs to the last tile that covers it
    print(f'  Phase 2: building ownership map...')
    ownership = np.full((ny, nx), -1, dtype=np.int32)
    for tile_id, (xi_lo, xi_hi, yi_lo, yi_hi) in enumerate(slices):
        if xi_hi > xi_lo and yi_hi > yi_lo:
            ownership[yi_lo:yi_hi, xi_lo:xi_hi][region.values[yi_lo:yi_hi, xi_lo:xi_hi]] = tile_id
    print(f'  Phase 2 done.')

    # Phase 3: generate transcripts per tile; keep only those in owned L2/3 pixels
    print(f'  Phase 3: generating transcripts for {len(positions)} tiles...')
    in_l23_orig = transcripts_in_region(df, region, xcol='global_x', ycol='global_y')
    result_df = df[~in_l23_orig].reset_index(drop=True)

    all_tiles = []
    for tile_id, ((rx0, ry0), _) in enumerate(zip(positions, slices)):
        if tile_id % 50 == 0:
            print(f'    tile {tile_id}/{len(positions)}', flush=True)
        rx1, ry1 = rx0 + PATCH_SIZE_UM, ry0 + PATCH_SIZE_UM

        tile = patch_df.copy()
        tile['global_x'] = tile['global_x'] + (rx0 - px0)
        tile['global_y'] = tile['global_y'] + (ry0 - py0)
        tile['x'] = tile['global_x']
        tile['y'] = tile['global_y']

        parts = [tile]
        for g_idx, g in enumerate(secret_genes_orig):
            n_add = rng.poisson(SECRET_MEAN_FRAC * n_patch)
            if n_add == 0:
                continue
            gx = rng.uniform(rx0, rx1, n_add)
            gy = rng.uniform(ry0, ry1, n_add)
            parts.append(pd.DataFrame({
                'Unnamed: 0':    -1,
                'barcode_id':    -1,
                'global_x':      gx,
                'global_y':      gy,
                'global_z':      0.0,
                'x':             gx,
                'y':             gy,
                'fov':           -1,
                'gene':          g,
                'transcript_id': [f'added_{tile_id}_{g_idx}_{k}' for k in range(n_add)],
                'cell_id':       -1,
            }))

        combined = pd.concat(parts, ignore_index=True)
        combined = combined[rng.uniform(0, 1, len(combined)) > DROPOUT_PROB]

        # Single vectorized ownership lookup — replaces both region clipping and overlap removal
        xi = np.round((combined['global_x'].values - l23_x0) / l23_dx).astype(int)
        yi = np.round((combined['global_y'].values - l23_y0) / l23_dy).astype(int)
        in_bounds = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        owned = np.zeros(len(combined), dtype=bool)
        owned[in_bounds] = ownership[yi[in_bounds], xi[in_bounds]] == tile_id
        combined = combined[owned].reset_index(drop=True)

        if len(combined) > 0:
            all_tiles.append(combined)

    n_placed = sum(len(t) for t in all_tiles)
    print(f'  Phase 3 done: {n_placed} transcripts placed across {len(all_tiles)} non-empty tiles.')
    if not all_tiles:
        return result_df
    return pd.concat([result_df] + all_tiles, ignore_index=True)


# ── 1. Select top-10 samples (top-25 by patch density, then top-10 by L2/3) ──

print('Loading cell metadata...')
cells = pd.read_csv(CELLS_FILE, sep='\t')
cells['sid'] = cells.Section.str.split('_').str[0:2].str.join('_')

# print(f'Reading patch counts from {H5AD_FILE}...')
# d = sc.read_h5ad(H5AD_FILE)
# d.obs.sid = d.obs.donor.astype(str) + '_' + d.obs.sid.astype(str)
# top25_sids = set(d.obs['sid'].value_counts().nlargest(25).index)
# print(f'Top 25 sids by patch count: {sorted(top25_sids)}')

# print('Computing L2/3 region sizes for top-25 candidates...')
# records = []
# for donor in sorted(os.listdir(RAW_DIR)):
#     donor_dir = os.path.join(RAW_DIR, donor)
#     if not os.path.isdir(donor_dir):
#         continue
#     for id_ in sorted(os.listdir(donor_dir)):
#         path = os.path.join(donor_dir, id_, 'cellpose-detected_transcripts.csv')
#         if not os.path.isfile(path):
#             continue
#         sid = f'{donor}_{id_}'
#         if sid not in top25_sids:
#             continue
#         sid_cells = cells[cells.sid == sid]
#         n_l23 = l23_region_size(sid_cells)
#         records.append({'donor': donor, 'id': id_, 'sid': sid, 'path': path, 'l23_pixels': n_l23})
#         print(f'  {sid}: {n_l23} L2/3 pixels')

# samples = pd.DataFrame(records).nlargest(10, 'l23_pixels').reset_index(drop=True)
# case_idx = rng.choice(10, size=5, replace=False)
# samples['status'] = 'control'
# samples.loc[case_idx, 'status'] = 'case'

_sm = pd.read_csv(os.path.join(os.path.dirname(OUT_DIR), 'samplemeta.tsv'), sep='\t')
samples = pd.DataFrame({
    'sid': _sm.sid,
    'status': _sm.case.map({1.0: 'case', 0.0: 'control'}),
})
print(f'Loaded {len(samples)} samples from samplemeta.tsv')

# ── 2. Identify SECRET genes via PCA on original unmodified data ──────────────

# print('\nRunning prepare_merfish + pca_pixels on original data to identify SECRET genes...')
# path_to_sid = dict(zip(samples.path, samples.sid))
#
#
# def load_seaad(path):
#     sid = path_to_sid[path]
#     df = pd.read_csv(path, dtype={9: str})
#     df = df[~df.gene.str.startswith('Blank')].reset_index(drop=True)
#     return sid, df
#
#
# tmpdir = tempfile.mkdtemp(prefix='vima_makedata_')
# try:
#     vima.pp.st.prepare_merfish(
#         load=load_seaad,
#         filepaths=list(path_to_sid.keys()),
#         x_col='global_x', y_col='global_y', gene_col='gene',
#         outdir=tmpdir,
#         pixel_size=RESOLUTION_UM,
#         basic_plots=False,
#     )
#     _, loadings = vima.pp.pca_pixels(tmpdir, 'metamarkers_10', nmetamarkers=10, plot=False)
# finally:
#     shutil.rmtree(tmpdir, ignore_errors=True)
#
# secret_genes_orig = loadings['PC2'].nlargest(N_SECRET).index.tolist()
# rename_map = {g: f'SECRET{i + 1}' for i, g in enumerate(secret_genes_orig)}
# print('SECRET genes (original names):', secret_genes_orig)

# ── 3. Choose 400×400µm patch from first case sample's L2/3 ─────────────────

# first_case = samples[samples.status == 'case'].iloc[0]
# print(f'\nSelecting patch from {first_case.sid}...')
#
# src_df = pd.read_csv(first_case.path, dtype={9: str})
# src_df = src_df[~src_df.gene.str.startswith('Blank')].reset_index(drop=True)
# src_cells = cells[cells.sid == first_case.sid]
# src_region = get_l23it_region(src_df, src_cells)
#
# yi, xi = find_patch_corner(src_region)
# px0 = float(src_region.x.values[xi])
# py0 = float(src_region.y.values[yi])
# px1 = px0 + PATCH_SIZE_UM
# py1 = py0 + PATCH_SIZE_UM
#
# in_patch = (
#     (src_df.global_x >= px0) & (src_df.global_x < px1) &
#     (src_df.global_y >= py0) & (src_df.global_y < py1)
# )
# patch_df = src_df[in_patch].copy().reset_index(drop=True)
# n_patch = len(patch_df)
# print(f'  Patch corner: ({px0:.1f}, {py0:.1f}) µm  |  {n_patch} transcripts')

# ── 4. Build output CSVs ──────────────────────────────────────────────────────

# Clear any previously processed data so stale normalized files don't accumulate
# and cause gene-count mismatches in pca_pixels.
st_dir = os.path.dirname(OUT_DIR)
# for subdir in ('normalized', 'masks'):
#     stale = os.path.join(st_dir, '10u', subdir)
#     if os.path.isdir(stale):
#         shutil.rmtree(stale)
#         print(f'Cleared stale directory: {stale}')

# os.makedirs(OUT_DIR, exist_ok=True)

# for _, row in samples.iterrows():
#     sid, path, status = row.sid, row.path, row.status
#     print(f'\nProcessing {sid} ({status})...')
#
#     df = pd.read_csv(path, dtype={9: str})
#     df = df[~df.gene.str.startswith('Blank')].reset_index(drop=True)
#
#     if status == 'case':
#         sid_cells = cells[cells.sid == sid]
#         region = get_l23it_region(df, sid_cells)
#         n_before = int(transcripts_in_region(df, region, xcol='global_x', ycol='global_y').sum())
#         df = tile_l23(df, region, patch_df, (px0, py0), secret_genes_orig, n_patch)
#         n_after = int(transcripts_in_region(df, region, xcol='global_x', ycol='global_y').sum())
#         print(f'  L2/3 transcripts: {n_before} original → {n_after} tiled')
#
#     df['gene'] = df['gene'].map(lambda g: rename_map.get(g, g))
#
#     out_path = os.path.join(OUT_DIR, f'{sid}.csv')
#     df.to_csv(out_path, index=False)
#     print(f'  Saved {len(df)} transcripts → {out_path}')

# ── 5. Metadata files ─────────────────────────────────────────────────────────

samplemeta = pd.DataFrame({
    'sid':   samples.sid,
    'donor': [chr(ord('A') + i) for i in range(len(samples))],
    'case':  (samples.status == 'case').astype(float),
}).reset_index(drop=True)
samplemeta_path = os.path.join(st_dir, 'samplemeta.tsv')
samplemeta.to_csv(samplemeta_path, sep='\t', index=False)
print(f'\nSaved samplemeta → {samplemeta_path}')

kept_sids = set(samples.sid)
cells_out = cells[cells.sid.isin(kept_sids)][['sid', 'x', 'y', 'subclass_name']].copy()
cells_out.index.name = 'cell_id'

secret_rows = []
for _, row in samples[samples.status == 'case'].iterrows():
    sid = row.sid
    sid_cells = cells[cells.sid == sid]
    n_l23it = int((sid_cells.subclass_name == 'L2/3 IT').sum())
    if n_l23it == 0:
        continue
    proxy_df = pd.DataFrame({
        'global_x': [sid_cells['x'].min(), sid_cells['x'].max()],
        'global_y': [sid_cells['y'].min(), sid_cells['y'].max()],
    })
    region = get_l23it_region(proxy_df, sid_cells)
    ys_idx, xs_idx = np.where(region.values)
    chosen = rng.choice(len(ys_idx), size=n_l23it, replace=True)
    secret_rows.append(pd.DataFrame({
        'sid': sid,
        'x': region.x.values[xs_idx[chosen]],
        'y': region.y.values[ys_idx[chosen]],
        'subclass_name': 'secret',
    }))
    print(f'  Added {n_l23it} secret cells for {sid}')

if secret_rows:
    secret_df = pd.concat(secret_rows, ignore_index=True)
    next_id = len(cells_out)
    secret_df.index = range(next_id, next_id + len(secret_df))
    secret_df.index.name = 'cell_id'
    cells_modified = pd.concat([cells_out, secret_df])
else:
    cells_modified = cells_out.copy()

cells_modified_path = os.path.join(st_dir, 'cells.tsv')
cells_modified.to_csv(cells_modified_path, sep='\t')
print(f'Saved cells_modified → {cells_modified_path}')

# ── 6. Archive ────────────────────────────────────────────────────────────────

print('Archiving...')
archive = os.path.join(HERE, 'ST_raw.tar.gz')
with tarfile.open(archive, 'w:gz') as tar:
    # tar.add(OUT_DIR, arcname='data/ST/raw')
    tar.add(samplemeta_path, arcname='data/ST/samplemeta.tsv')
    tar.add(cells_modified_path, arcname='data/ST/cells_modified.tsv')
print(f'Archived → {archive}')

print('Done.')
