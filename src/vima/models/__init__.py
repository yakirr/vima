from .resnet_vae import ResnetVAE
from .simple_vae import SimpleVAE
import torch

def cVAE(nmarkers, nsamples, nreps=10, compile_mode='default', weights=None):
    if torch.cuda.is_available() and compile_mode is not None:
        models = torch.nn.ModuleList([
                torch.compile(ResnetVAE(nmarkers, nsamples), mode=compile_mode)
            for _ in range(nreps)])
    else:
        models = torch.nn.ModuleList([
                ResnetVAE(nmarkers, nsamples)
            for _ in range(nreps)])
    if weights is not None:
        models.load_state_dict(torch.load(weights))
    return models

__all__ = ["ResnetVAE", "SimpleVAE"]