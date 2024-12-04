from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import pandas as pd
import random
import torch
from . import samples as tds
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
    def __init__(self, patchmeta, samples, sid_nums=None, standardize=True, percentile_thresh=99):
        self.samples = samples
        self.meta = patchmeta
        self.nchannels = next(iter(samples.values())).sizes['marker']

        self.pytorch_mode()
        self.__preprocess__(standardize, percentile_thresh, sid_nums=sid_nums)
        self.augmentation_off()

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