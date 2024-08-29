from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import pandas as pd
import random
import torch

class DiscreteRotation:
    def __init__(self, angles):
        self.angles = angles
    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class ToTensor:
    def __call__(self, x):
        return torch.tensor(x).permute(2,0,1)

random_transform = transforms.Compose([
    ToTensor(),
    DiscreteRotation(angles=[0,90,180,270]),
    transforms.RandomHorizontalFlip()
])

all_transforms = [
    transforms.Compose([
        ToTensor(),
        DiscreteRotation(angles=[0]),
    ]),
    transforms.Compose([
        ToTensor(),
        DiscreteRotation(angles=[90]),
    ]),
    transforms.Compose([
        ToTensor(),
        DiscreteRotation(angles=[180]),
    ]),
    transforms.Compose([
        ToTensor(),
        DiscreteRotation(angles=[270]),
    ]),
    transforms.Compose([
        ToTensor(),
        DiscreteRotation(angles=[0]),
        transforms.RandomHorizontalFlip(p=1),
    ]),
    transforms.Compose([
        ToTensor(),
        DiscreteRotation(angles=[90]),
        transforms.RandomHorizontalFlip(p=1),
    ]),
    transforms.Compose([
        ToTensor(),
        DiscreteRotation(angles=[180]),
        transforms.RandomHorizontalFlip(p=1),
    ]),
    transforms.Compose([
        ToTensor(),
        DiscreteRotation(angles=[270]),
        transforms.RandomHorizontalFlip(p=1),
    ]),
]

def choose_patches(masks, patchsize, patchstride, extract_donor, max_frac_empty=0.2):
    patch_meta = []

    for sid, mask in masks.items():
        starts = np.array([
            [x, y]
            for x in range(0, mask.shape[0]-patchsize, patchstride)
            for y in range(0, mask.shape[1]-patchsize, patchstride)
            if np.mean(mask.values[x:x+patchsize,y:y+patchsize]) > (1-max_frac_empty)
        ]).astype('int')

        donor = extract_donor(sid)
        patch_meta.append(pd.DataFrame([
                (sid, donor, x, y, mask.index[x], mask.columns[y])
                for x, y in starts
            ],
            columns=['id','donor','x','y', 'x_microns', 'y_microns'],
        ))
    patch_meta = pd.concat(patch_meta, axis=0).reset_index(drop=True)
    patch_meta.x = patch_meta.x.astype('int')
    patch_meta.y = patch_meta.y.astype('int')
    patch_meta.x_microns = patch_meta.x_microns.astype('float32')
    patch_meta.y_microns = patch_meta.y_microns.astype('float32')
    return patch_meta

class PatchCollection(Dataset):
    def __init__(self, patch_meta, samples, masks, patchsize, normalize='global'):
        self.samples = samples
        self.masks = masks
        self.meta = patch_meta
        self.patchsize = patchsize
        self.nchannels = next(iter(samples.values())).shape[-1]

        self.dim_order = 'pytorch'
        self.augment()

        self.__preprocess__(normalize)
        self.pmin = np.min([np.percentile(s, 1, axis=(0,1)) for s in self.samples.values()], axis=0)
        self.pmax = np.max([np.percentile(s, 99, axis=(0,1)) for s in self.samples.values()], axis=0)

    def augment(self):
        self.transform = random_transform
    def no_augment(self):
        self.transform = random_transform[0]

    def __preprocess__(self, style):
        if style == 'global':
            allpixels = np.concatenate([
                s[m] for s, m in zip(self.samples.values(), self.masks.values())])

            self.means = allpixels.mean(axis=0, dtype='float64')
            self.stds = allpixels.std(axis=0, dtype='float64')
            del allpixels

            print(f'means: {self.means}')
            print(f'stds: {self.stds}')
            for sid in self.samples.keys():
                self.samples[sid][self.masks[sid].values] -= self.means
                self.samples[sid][self.masks[sid].values] /= self.stds
        elif style == 'persample':
            allpixels = {
                sid: self.samples[sid][self.masks[sid]] for sid in self.samples.keys()
            }
            self.means = {
                sid: allpixels[sid].mean(axis=0, dtype='float64') for sid in self.samples.keys()
            }
            self.stds = {
                sid: allpixels[sid].std(axis=0, dtype='float64') for sid in self.samples.keys()
            }
            del allpixels

            print(f'means: {self.means}')
            print(f'stds: {self.stds}')
            for sid in self.samples.keys():
                self.samples[sid] -= self.means[sid]
                self.samples[sid] /= self.stds[sid]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        toget = self.meta.iloc[idx]

        if type(idx) == int:
            patch = self.samples[toget.id][toget.x:toget.x+self.patchsize,toget.y:toget.y+self.patchsize]
            if self.dim_order == 'numpy':
                return patch
            else:
                return self.transform(patch)
        else:
            patches = np.array([
                self.samples[s][x:x+self.patchsize,y:y+self.patchsize]
                for s, x, y in toget[['id','x','y']].values])
            if self.dim_order == 'numpy':
                return patches
            else:
                return torch.stack([self.transform(p) for p in patches])