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
    def choose_patches(samples, patchsize, patchstride, max_frac_empty):
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
                    (s.sid, s.donor, i, j, mask.x[i], mask.y[j])
                    for i, j in starts
                ],
                columns=['sid','donor','x','y', 'x_microns', 'y_microns'],
            ))
        patchmeta = pd.concat(patchmeta, axis=0).reset_index(drop=True)
        patchmeta.x = patchmeta.x.astype('int')
        patchmeta.y = patchmeta.y.astype('int')
        patchmeta.x_microns = patchmeta.x_microns.astype('float32')
        patchmeta.y_microns = patchmeta.y_microns.astype('float32')
        patchmeta['patchsize'] = patchsize
        return patchmeta

    def __init__(self, samples, patchsize=40, patchstride=10, max_frac_empty=0.8,
                sid_nums=None, standardize=True, percentile_thresh=99):
        self.samples = samples
        self.meta = PatchCollection.choose_patches(samples, patchsize, patchstride, max_frac_empty)
        self.nmarkers = next(iter(samples.values())).sizes['marker']

        self.pytorch_mode()
        self.__preprocess__(standardize, percentile_thresh, sid_nums=sid_nums)
        self.augmentation_off()

    @property
    def sid_nums(self):
        return {sid:sid_num for sid, sid_num in self.meta[['sid','sid_num']].drop_duplicates().values}

    @property
    def nsamples(self):
        return len(self.meta.sid.unique())

    def augmentation_on(self):
        if self.dim_order != 'pytorch':
            print('Data augmentation only available in pytorch mode. Will leave augmentation off')
            return
        print('data augmentation is on')
        self.transform = transforms.Compose([
            ToTorch(),
            RandomDiscreteRotation(),
            transforms.RandomHorizontalFlip(),
            ])
    def augmentation_off(self):
        print('data augmentation is off')
        self.transform = transforms.Compose([
            ToTorch(),
            ])

    def pytorch_mode(self):
        self.dim_order = 'pytorch'
        print('in pytorch mode')
    def numpy_mode(self):
        self.dim_order = 'numpy'
        self.augmentation_off()
        print('in numpy mode')

    def __preprocess__(self, standardize, percentile_thresh, sid_nums=None):
        self.patches = np.array([
            self.samples[s].data[y:y+ps,x:x+ps,:]
            for s, x, y, ps in self.meta[['sid','x','y','patchsize']].values
            ])
        if sid_nums is None:
            self.meta['sid_num'] = pd.factorize(self.meta.sid)[0]
        else:
            self.meta['sid_num'] = self.meta.sid.map(sid_nums)

        ix = np.random.choice(len(self), min(50000, len(self)), replace=False)
        subset = self.patches[ix]
        self.means = subset.mean(axis=(0,1,2))
        self.stds = subset.std(axis=(0,1,2))
        self.percentiles = np.percentile(np.abs(subset), percentile_thresh, axis=(0,1,2))
        self.vmin = (-self.means - self.percentiles)/self.stds
        self.vmax = (-self.means + self.percentiles)/self.stds
        print(f'means: {self.means}')
        print(f'stds: {self.stds}')

        if standardize:
            self.patches = (self.patches - self.means[None,None,None,:]) / self.stds[None,None,None,:]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        patches = self.patches[idx]
        sid_nums = self.meta.sid_num.values[idx]
        if self.dim_order == 'numpy':
            return patches, sid_nums
        else:
            return self.transform(patches), torch.tensor(sid_nums)