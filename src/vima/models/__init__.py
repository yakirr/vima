from .resnet_vae import ResnetVAE
from .simple_vae import SimpleVAE
import torch

def cVAE(nmarkers, nsamples, nreps=10, compile_mode=None):
    if torch.cuda.is_available() and compile_mode is not None:
        return torch.nn.ModuleList([
                torch.compile(ResnetVAE(nmarkers, nsamples), mode=compile_mode)
            for _ in range(nreps)])
    else:
        return torch.nn.ModuleList([
                ResnetVAE(nmarkers, nsamples)
            for _ in range(nreps)])

__all__ = ["ResnetVAE", "SimpleVAE"]