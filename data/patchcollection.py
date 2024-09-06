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

class PatchCollection(Dataset):
    def __init__(self, patchmeta, samples, standardize=True):
        self.samples = samples
        self.meta = patchmeta
        self.nchannels = next(iter(samples.values())).sizes['marker']

        self.pytorch_mode()
        self.transform = all_transforms[0]
        self.__preprocess__(standardize)
        self.augmentation_off()

    def augmentation_on(self):
        if self.dim_order != 'pytorch':
            print('Data augmentation only available in pytorch mode. Will leave augmentation off')
            return
        print('data augmentation is on')
        self.transform = transforms.Compose([
            random_transform,
            self.scale])
    def augmentation_off(self):
        print('data augmentation is off')
        self.transform = transforms.Compose([
            all_transforms[0],
            self.scale])
    def pytorch_mode(self):
        self.dim_order = 'pytorch'
        print('in pytorch mode')
    def numpy_mode(self):
        self.dim_order = 'numpy'
        print('in numpy mode')

    def __preprocess__(self, standardize):
        ix = np.random.choice(len(self), min(10000, len(self)), replace=False)
        subset = self[ix].cpu().numpy()
        self.means = subset.mean(axis=(0,2,3))
        self.stds = subset.std(axis=(0,2,3))
        percentiles = np.percentile(np.abs(subset), 99, axis=(0,2,3))
        self.vmin = -self.means - percentiles
        self.vmax = -self.means + percentiles
        print(f'means: {self.means}')
        print(f'stds: {self.stds}')

        if standardize:
            self.scale = transforms.Normalize(mean=self.means, std=self.stds)
        else:
            self.scale = transforms.Normalize(mean=np.zeros(self.nchannels), std=np.ones(self.nchannels))        

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        toget = self.meta.iloc[idx]

        if type(idx) == int:
            patch = self.samples[toget.sid].data[toget.y:toget.y+toget.patchsize,toget.x:toget.x+toget.patchsize]
            if self.dim_order == 'numpy':
                return patch
            else:
                return self.transform(patch)
        else:
            patches = np.array([
                self.samples[s].data[y:y+ps,x:x+ps]
                for s, x, y, ps in toget[['sid','x','y', 'patchsize']].values])
            if self.dim_order == 'numpy':
                return patches
            else:
                return torch.stack([self.transform(p) for p in patches])

class RandomDiscreteRotation:
    def __init__(self, angles):
        self.angles = angles
    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class ToTorch:
    def __call__(self, x):
        return torch.tensor(x).permute(2,0,1)

random_transform = transforms.Compose([
    ToTorch(),
    RandomDiscreteRotation(angles=[0,90,180,270]),
    transforms.RandomHorizontalFlip()
])

all_transforms = [
    transforms.Compose([
        ToTorch(),
    ]),
    transforms.Compose([
        ToTorch(),
        RandomDiscreteRotation(angles=[90]),
    ]),
    transforms.Compose([
        ToTorch(),
        RandomDiscreteRotation(angles=[180]),
    ]),
    transforms.Compose([
        ToTorch(),
        RandomDiscreteRotation(angles=[270]),
    ]),
    transforms.Compose([
        ToTorch(),
        transforms.RandomHorizontalFlip(p=1),
    ]),
    transforms.Compose([
        ToTorch(),
        RandomDiscreteRotation(angles=[90]),
        transforms.RandomHorizontalFlip(p=1),
    ]),
    transforms.Compose([
        ToTorch(),
        RandomDiscreteRotation(angles=[180]),
        transforms.RandomHorizontalFlip(p=1),
    ]),
    transforms.Compose([
        ToTorch(),
        RandomDiscreteRotation(angles=[270]),
        transforms.RandomHorizontalFlip(p=1),
    ]),
]