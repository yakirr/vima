import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, '../../../../ST/ALZ/alz-data/transcripts')
CELLS_FILE = os.path.join(HERE, '../../../../ST/ALZ/alz-data/SEAAD_MTG_MERFISH_metadata.2024-05-03.noblanks.harmonized.txt')
OUT_DIR = os.path.join(HERE, 'ST/raw')


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

# 5 & 6. Load each sample, spike in SECRET for case samples, save without Blank genes
os.makedirs(OUT_DIR, exist_ok=True)

for _, row in samples.iterrows():
    sid, path, status = row.sid, row.path, row.status
    print(f'\nProcessing {sid} ({status})...')

    df = pd.read_csv(path, dtype={9: str})
    df = df[~df.gene.str.startswith('Blank')].reset_index(drop=True)

    sid_cells = cells[cells.sid == sid]
    spike_rows = []

    if status == 'case':
        l23it = sid_cells[sid_cells.subclass_name == 'L2/3 IT']
        print(f'  Spiking into {len(l23it)} L2/3 IT cells')
        spike_rows += [(c.x + rng.normal(10, 10, 300), c.y + rng.normal(10, 10, 300))
                       for _, c in l23it.iterrows()]

    chosen = sid_cells.sample(n=min(100, len(sid_cells)), random_state=rng.integers(2**31))
    spike_rows += [(c.x + rng.normal(10, 10, 5), c.y + rng.normal(10, 10, 5))
                   for _, c in chosen.iterrows()]

    spike_x = np.concatenate([xs for xs, _ in spike_rows])
    spike_y = np.concatenate([ys for _, ys in spike_rows])

    spike_df = pd.DataFrame({
        'Unnamed: 0':    -1,
        'barcode_id':    0,
        'global_x':      spike_x,
        'global_y':      spike_y,
        'global_z':      0.0,
        'x':             spike_x,
        'y':             spike_y,
        'fov':           -1,
        'gene':          'SECRET',
        'transcript_id': [f'spike_{i}' for i in range(len(spike_x))],
        'cell_id':       -1,
    })
    df = pd.concat([df, spike_df], ignore_index=True)
    print(f'  Added {len(spike_df)} SECRET transcripts')

    secret_mask = df.gene == 'SECRET'
    df.loc[secret_mask, 'gene'] = rng.choice(
        [f'SECRET{i}' for i in range(1, 31)], size=secret_mask.sum())

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
import tarfile
archive = os.path.join(HERE, 'ST_raw.tar.gz')
print(f'\nArchiving {OUT_DIR} → {archive}')
with tarfile.open(archive, 'w:gz') as tar:
    tar.add(OUT_DIR, arcname='data/raw')
print('Done.')
