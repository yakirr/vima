import glob
import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv2
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)

def read_samples(files, stop_after=None):
    if type(files) == str:
        files = glob.glob(files)
    if stop_after is None: stop_after = len(files)

    samples = {}
    for f in pb(files[:stop_after]):
        s = xr.open_dataarray(f).astype(np.float32)
        s.attrs['sid'] = os.path.splitext(os.path.basename(f))[0]
        samples[s.sid] = s

    return samples

def reindex_by_sid(samplemeta, sid_to_donor):
    # change samplemeta so that each row is a sample rather than a donor
    sids = sid_to_donor.keys()
    in_our_data = pd.DataFrame({
        'sid': sids,
        'donor': [sid_to_donor[sid] for sid in sids]
    })
    return pd.merge(in_our_data, samplemeta, left_on='donor', right_index=True, how='left').set_index('sid', drop=True)

def get_mask(s):
    return (s!=0).any(dim='marker')

def union_patches_in_sample(patchmeta, s):
    res = s[:,:,0].copy()
    res[:,:] = 0

    for _, p in patchmeta[patchmeta.sid == s.sid].iterrows():
        res[p.y:p.y+p.patchsize, p.x:p.x+p.patchsize] = 1

    return res