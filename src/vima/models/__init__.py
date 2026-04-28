from .resnet_vae import ResnetVAE
from .simple_vae import SimpleVAE
import torch

def cVAE(nmarkers, covariate_sizes, nreps=10, compile_mode='default', weights=None):
    if torch.cuda.is_available() and compile_mode is not None:
        models = torch.nn.ModuleList([
                torch.compile(ResnetVAE(nmarkers, covariate_sizes), mode=compile_mode)
            for _ in range(nreps)])
    else:
        models = torch.nn.ModuleList([
                ResnetVAE(nmarkers, covariate_sizes)
            for _ in range(nreps)])
    if weights is not None:
        if torch.cuda.is_available():
            map_location = None
            uncompile = False
        elif torch.backends.mps.is_available():
            map_location = torch.device('mps')
            uncompile = True
        else:
            map_location = torch.device('cpu')
            uncompile = True
        state = torch.load(weights, map_location=map_location, weights_only=True)

        if uncompile:
            state = {k.replace('._orig_mod', ''): v for k, v in state.items()}
        models.load_state_dict(state)
    return models

__all__ = ["ResnetVAE", "SimpleVAE"]