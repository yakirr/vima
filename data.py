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

def choose_patches(masks, patchsize, patchstride, extract_donor):
    patch_meta = []

    for sid, mask in masks.items():
        starts = np.array([
            [x, y]
            for x in range(0, mask.shape[0]-patchsize, patchstride)
            for y in range(0, mask.shape[1]-patchsize, patchstride)
            if np.mean(mask[x:x+patchsize,y:y+patchsize]) > 0.8
        ]).astype('int')

        donor = extract_donor(sid)
        patch_meta.append(pd.DataFrame([
                (sid, donor, x, y)
                for x, y in starts
            ],
            columns=['id','donor','x','y'],
        ))
    patch_meta = pd.concat(patch_meta, axis=0).reset_index(drop=True)
    patch_meta.x = patch_meta.x.astype('int')
    patch_meta.y = patch_meta.y.astype('int')
    return patch_meta

class ConcretePatchCollection(Dataset):
    def __init__(self, patches, patch_meta, transform=None):
        self.patches = patches
        self.meta = patch_meta
        self.patchsize = patches[0].shape[0]
        self.nchannels = patches[0].shape[-1]

        if transform is None:
            self.mode = 'numpy'
            self.transform = None
        else:
            self.mode = 'pytorch'
            self.transform = transform

        self.pmin = np.min(self.patches, axis=(0,1,2))
        self.pmax = np.max(self.patches, axis=(0,1,2))

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if type(idx) == int:
            if self.mode == 'numpy':
                return self.patches[idx]
            else:
                return self.transform(
                    self.patches[idx]
                )
        else:
            if self.mode == 'numpy':
                return self.patches[idx]
            else:
                return torch.stack([self.transform(p) for p in self.patches[idx]])

class PatchCollection(Dataset):
    def __init__(self, patch_meta, samples, masks, patchsize, normalize='global', transform=None):
        self.samples = samples
        self.masks = masks
        self.meta = patch_meta
        self.patchsize = patchsize
        self.nchannels = next(iter(samples.values())).shape[-1]
        
        if transform is None:
            self.mode = 'numpy'
            self.transform = None
        else:
            self.mode = 'pytorch'
            self.transform = transform

        if normalize is not False:
            self.__preprocess__(normalize)

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
                self.samples[sid] -= self.means
                self.samples[sid] /= self.stds
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
            print(f'means: {self.means}')
            print(f'stds: {self.stds}')
            for sid in self.samples.keys():
                self.samples[sid] -= self.means[sid]
                self.samples[sid] /= self.stds[sid]
        self.pmin = np.min([np.percentile(s, 1, axis=(0,1)) for s in self.samples.values()], axis=0)
        self.pmax = np.max([np.percentile(s, 99, axis=(0,1)) for s in self.samples.values()], axis=0)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        toget = self.meta.iloc[idx]

        if type(idx) == int:
            if self.mode == 'numpy':
                return self.samples[toget.id][toget.x:toget.x+self.patchsize,toget.y:toget.y+self.patchsize]
            else:
                return self.transform(
                    self.samples[toget.id][toget.x:toget.x+self.patchsize,toget.y:toget.y+self.patchsize]
                )
        else:
            patches = np.array([
                self.samples[s][x:x+self.patchsize,y:y+self.patchsize]
                for s, x, y in toget[['id','x','y']].values])
            if self.mode == 'numpy':
                return patches
            else:
                return torch.stack([self.transform(p) for p in patches])