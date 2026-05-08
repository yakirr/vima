import os
import numpy as np
import pandas as pd
import xarray as xr
import cv2

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
RESOLUTION_UM = 10

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, '../../../../ST/ALZ/alz-data/transcripts')
CELLS_FILE = os.path.join(HERE, '../../../../ST/ALZ/alz-data/SEAAD_MTG_MERFISH_metadata.2024-05-03.noblanks.harmonized.txt')
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


def make_spike_transcripts(cells_in_reg, gene_pool, n_per_cell=300):
    """For each cell in the region place n_per_cell SECRET transcripts in a
    Gaussian (stddev=10 µm) around the cell centroid."""
    if len(cells_in_reg) == 0:
        return pd.DataFrame(columns=[
            'Unnamed: 0', 'barcode_id', 'global_x', 'global_y', 'global_z',
            'x', 'y', 'fov', 'gene', 'transcript_id', 'cell_id',
        ])

    spike_x = np.concatenate([rng.normal(cx, 10, n_per_cell) for cx in cells_in_reg['x'].values])
    spike_y = np.concatenate([rng.normal(cy, 10, n_per_cell) for cy in cells_in_reg['y'].values])
    n = len(spike_x)

    return pd.DataFrame({
        'Unnamed: 0':    -1,
        'barcode_id':    0,
        'global_x':      spike_x,
        'global_y':      spike_y,
        'global_z':      0.0,
        'x':             spike_x,
        'y':             spike_y,
        'fov':           -1,
        'gene':          rng.choice(gene_pool, size=n),
        'transcript_id': [f'spike_{i}' for i in range(n)],
        'cell_id':       -1,
    })


# 1. Collect all (donor, id) pairs and count transcripts via line count (fast)
records = []
for donor in sorted(os.listdir(RAW_DIR)):
    donor_dir = os.path.join(RAW_DIR, donor)
    if not os.path.isdir(donor_dir):
        continue
    for id_ in sorted(os.listdir(donor_dir)):
        path = os.path.join(donor_dir, id_, 'cellpose-detected_transcripts.csv')
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            n = sum(1 for _ in f) - 1  # subtract header
        records.append({'donor': donor, 'id': id_, 'sid': f'{donor}_{id_}', 'path': path, 'n': n})

samples = pd.DataFrame(records)

# Pick one sample per donor (the one with most transcripts)
samples = samples.sort_values('n', ascending=False).groupby('donor', sort=False).first().reset_index()
print(f'{len(samples)} donors found')

# 2. Keep top 10 by transcript count
samples = samples.nlargest(10, 'n').reset_index(drop=True)
print('Top 10 samples:')
print(samples[['sid', 'n']].to_string(index=False))

# 3. Randomly assign case/control
case_idx = rng.choice(10, size=5, replace=False)
samples['status'] = 'control'
samples.loc[case_idx, 'status'] = 'case'
case_sids = set(samples.loc[samples.status == 'case', 'sid'])
print('\nCase samples:', sorted(case_sids))

# 4. Load cells
print('\nLoading cells metadata...')
cells = pd.read_csv(CELLS_FILE, sep='\t')
cells['sid'] = cells.Section.str.split('_').str[0:2].str.join('_')

# 5. For each sample: remove transcripts in L2/3 IT region, replace with SECRET
os.makedirs(OUT_DIR, exist_ok=True)
relabeled_indices = []  # original cells-dataframe row indices whose type becomes 'secret'

for _, row in samples.iterrows():
    sid, path, status = row.sid, row.path, row.status
    print(f'\nProcessing {sid} ({status})...')

    df = pd.read_csv(path, dtype={9: str})
    df = df[~df.gene.str.startswith('Blank')].reset_index(drop=True)

    sid_cells = cells[cells.sid == sid]

    # Define L2/3 IT-rich region
    region = get_l23it_region(df, sid_cells)
    n_region_pixels = int(region.values.sum())
    print(f'  Region area: {n_region_pixels} pixels ({n_region_pixels * RESOLUTION_UM**2 / 1e6:.2f} mm²)')

    if n_region_pixels == 0:
        print(f'  WARNING: empty region for {sid}, skipping spike-in')
        df.to_csv(os.path.join(OUT_DIR, f'{sid}.csv'), index=False)
        continue

    # Find all cells (any type) whose centroid falls in the region
    in_reg_cells = transcripts_in_region(sid_cells, region)
    cells_in_reg = sid_cells[in_reg_cells]
    relabeled_indices.extend(cells_in_reg.index.tolist())
    print(f'  {len(cells_in_reg)} cells in region ({in_reg_cells.sum()} / {len(sid_cells)} total cells)')

    # Remove all transcripts in the region
    in_region = transcripts_in_region(df, region, xcol='global_x', ycol='global_y')
    n_removed = int(in_region.sum())
    df = df[~in_region].reset_index(drop=True)
    print(f'  Removed {n_removed} transcripts from region')

    # Replace with 300 SECRET transcripts per cell, Gaussian (stddev=10 µm) around each centroid
    gene_pool = [f'SECRET{i}' for i in range(1, 21)] if status == 'case' \
        else [f'SECRET{i}' for i in range(21, 41)]
    spike_df = make_spike_transcripts(cells_in_reg, gene_pool)
    df = pd.concat([df, spike_df], ignore_index=True)
    print(f'  Added {len(spike_df)} SECRET transcripts ({len(cells_in_reg)} cells × 300)')

    out_path = os.path.join(OUT_DIR, f'{sid}.csv')
    df.to_csv(out_path, index=False)
    print(f'  Saved {len(df)} transcripts → {out_path}')

# Create sample metadata file with case/ctrl status
samplemeta = pd.DataFrame({
    'sid':   samples.sid,
    'donor': samples.sid.str.split('_').str[0],
    'case':  (samples.status == 'case').astype(float),
}).reset_index(drop=True)
samplemeta_path = os.path.join(os.path.dirname(OUT_DIR), 'samplemeta.tsv')
samplemeta.to_csv(samplemeta_path, sep='\t', index=False)
print(f'\nSaved samplemeta → {samplemeta_path}')

# zip the transcript files for easy upload/download later
print('Zipping...')
import tarfile
archive = os.path.join(HERE, 'ST_raw.tar.gz')
print(f'\nArchiving {OUT_DIR} → {archive}')
with tarfile.open(archive, 'w:gz') as tar:
    tar.add(OUT_DIR, arcname='data/raw')

# Write filtered cell metadata for the selected samples
kept_sids = set(samples.sid)
cells_out = cells[cells.sid.isin(kept_sids)][['sid', 'x', 'y', 'subclass_name']].copy()
cells_out.index.name = 'cell_id'
cells_path = os.path.join(os.path.dirname(OUT_DIR), 'cells.tsv')
cells_out.to_csv(cells_path, sep='\t')
print(f'Saved cells → {cells_path}')

# Write cells_modified: same as cells but with all region cells relabeled as 'secret'
cells_modified = cells_out.copy()
cells_modified.loc[cells_modified.index.isin(relabeled_indices), 'subclass_name'] = 'secret'
cells_modified_path = os.path.join(os.path.dirname(OUT_DIR), 'cells_modified.tsv')
cells_modified.to_csv(cells_modified_path, sep='\t')
n_relabeled = cells_modified['subclass_name'].eq('secret').sum()
print(f'Saved cells_modified → {cells_modified_path} ({n_relabeled} cells relabeled as secret)')

print('Done.')
