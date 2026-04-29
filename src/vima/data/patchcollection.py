from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import pandas as pd
import random
import torch
from . import samples as vds
from tqdm import tqdm
pb = lambda x: tqdm(x, ncols=100)

class ToTorch:
    def __call__(self, x):
        return torch.tensor(x).permute(*range(x.ndim - 3), x.ndim-1, x.ndim-3, x.ndim-2)

class RandomDiscreteRotation:
    def __call__(self, x):
        ntimes = np.random.choice([0,1,2,3])
        for i in range(ntimes):
            x = torch.rot90(x, dims=[-2,-1])
        return x

class PatchCollection(Dataset):
    @staticmethod
    def choose_patches(samples, patchsize, patchstride, max_frac_empty, verbose=False):
        patchmeta = []

        for s in pb(samples.values()):
            mask = vds.get_mask(s)
            starts = np.array([
                [i, j]
                for i in range(0, mask.sizes['x']-patchsize, patchstride)
                for j in range(0, mask.sizes['y']-patchsize, patchstride)
                if mask.data[j:j+patchsize, i:i+patchsize].mean() > (1-max_frac_empty)
            ]).astype('int')

            patchmeta.append(pd.DataFrame([
                    (s.sid, i, j, mask.x[i], mask.y[j])
                    for i, j in starts
                ],
                columns=['sid','x','y', 'x_microns', 'y_microns'],
            ))
        patchmeta = pd.concat(patchmeta, axis=0).reset_index(drop=True)
        patchmeta.x = patchmeta.x.astype('int')
        patchmeta.y = patchmeta.y.astype('int')
        patchmeta.x_microns = patchmeta.x_microns.astype('float32')
        patchmeta.y_microns = patchmeta.y_microns.astype('float32')
        patchmeta['patchsize'] = patchsize
        return patchmeta
    
    def make_patchmeta(self, covariates=None, condition_on_sid=True):
        self.patches = np.array([
            self.samples[s].data[y:y+ps,x:x+ps,:]
            for s, x, y, ps in self.meta[['sid','x','y','patchsize']].values
            ])
        self.meta['sid_num'] = pd.factorize(self.meta.sid)[0]
        if condition_on_sid:
            self._covariate_cols.append('sid_num')
        if covariates:
            for name, mapping in covariates.items():
                col = f'{name}_num'
                self.meta[col] = pd.factorize(self.meta.sid.map(mapping))[0]
                self._covariate_cols.append(col)

    def compute_stats(self, percentile_thresh, verbose=False):
        ix = np.random.choice(len(self), min(50000, len(self)), replace=False)
        subset = self.patches[ix]
        self.means = subset.mean(axis=(0,1,2), dtype=np.float64).astype(np.float32)
        self.stds = subset.std(axis=(0,1,2), dtype=np.float64).astype(np.float32)
        self.percentiles = np.percentile(np.abs(subset), percentile_thresh, axis=(0,1,2))
        self.vmin = (-self.means - self.percentiles)/self.stds
        self.vmax = (-self.means + self.percentiles)/self.stds

        if verbose:
            fmt = lambda a: '  '.join(f'{v:.2g}' for v in a)
            print(f'per-channel means: {fmt(self.means)}')
            print(f'per-channel stds:  {fmt(self.stds)}')

    def standardize(self, verbose=False):
        if verbose: print('Standardizing patches...')
        self.patches = (self.patches - self.means[None,None,None,:]) / self.stds[None,None,None,:]

    def __init__(self, samples, patchsize=40, patchstride=10, max_frac_empty=0.8,
                standardize=True, percentile_thresh=99, verbose=False,
                covariates=None, condition_on_sid=True):
        self.samples = samples
        self.patchstride = patchstride
        self._covariate_cols = []
        self.meta = PatchCollection.choose_patches(samples, patchsize, patchstride, max_frac_empty, verbose=verbose)
        self.nmarkers = next(iter(samples.values())).sizes['marker']

        self.pytorch_mode()
        self.make_patchmeta(covariates=covariates, condition_on_sid=condition_on_sid)
        self.compute_stats(percentile_thresh, verbose=verbose)
        if standardize:
            self.standardize(verbose=verbose)
        self.augmentation_off()

    def refined(self, max_frac_empty, tol=1e-10, standardize=True, percentile_thresh=99,
                verbose=False):
        import copy
        empty_val = -self.means / self.stds
        empty_frac = (np.abs(self.patches - empty_val[None,None,None,:]).max(axis=-1) < tol).mean(axis=(1, 2))
        keep = np.where(empty_frac < max_frac_empty)[0]
        result = copy.copy(self)
        result.subset(keep)
        result.compute_stats(percentile_thresh, verbose=verbose)
        if standardize:
            result.standardize(verbose=verbose)
        return result

    @property
    def sid_nums(self):
        return {sid:sid_num for sid, sid_num in self.meta[['sid','sid_num']].drop_duplicates().values}

    @property
    def nsamples(self):
        return len(self.meta.sid.unique())

    @property
    def covariate_sizes(self):
        return [self.meta[col].nunique() for col in self._covariate_cols]

    def augmentation_on(self):
        if self.dim_order != 'pytorch':
            print('WARNING: Data augmentation only available in pytorch mode. Will leave augmentation off')
            return
        print('\033[90m[PatchCollection: data augmentation on]\033[0m')
        self.transform = transforms.Compose([
            ToTorch(),
            RandomDiscreteRotation(),
            transforms.RandomHorizontalFlip(),
            ])
    def augmentation_off(self):
        print('\033[90m[PatchCollection: data augmentation is off]\033[0m')
        self.transform = transforms.Compose([
            ToTorch(),
            ])

    def add_donor_ids(self, donor_ids_series):
        self.meta['donor'] = self.meta.sid.map(donor_ids_series)

    def pytorch_mode(self):
        self.dim_order = 'pytorch'
        print('\033[90m[PatchCollection: in pytorch mode]\033[0m')
    def numpy_mode(self):
        self.dim_order = 'numpy'
        self.augmentation_off()
        print('\033[90m[PatchCollection: in numpy mode]\033[0m')

    def subset(self, ix):
        self.patches = self.patches[ix]
        self.meta = self.meta.iloc[ix]

    def __repr__(self):
        ps = self.meta.patchsize.iloc[0]
        cov_parts = [f'{col.removesuffix("_num")} ({self.meta[col].nunique()} values)'
                     for col in self._covariate_cols]
        cov_str = ', '.join(cov_parts) if cov_parts else 'none'
        return (
            f'PatchCollection object with npatches × width × height × nmarkers = {len(self)}×{ps}×{ps}×{self.nmarkers}\n'
            f'\tcovariates: {cov_str}'
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        patches = self.patches[idx]
        covars = self.meta[self._covariate_cols].values[idx]
        if self.dim_order == 'numpy':
            return patches, covars
        else:
            return self.transform(patches), torch.tensor(covars, dtype=torch.long)