from .resnet_vae import ResnetVAE
from .simple_vae import SimpleVAE

import torch
def cVAE(nmarkers, nsamples, nreps=3):
    return torch.nn.ModuleList([
            ResnetVAE(nmarkers, nsamples)
        for _ in range(nreps)])

__all__ = ["ResnetVAE", "SimpleVAE"]