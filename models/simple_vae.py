import numpy as np
import torch
from torch import nn, Tensor
from .vae import VAE

class SimpleVAE(VAE):
    """Simple convolutional variational autoencoder."""

    def __init__(self, ncolors : int, patch_size : int,
            latent_dim: int=100, nfilters1: int=256, nfilters2: int=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.nfilters1 = nfilters1
        self.nfilters2 = nfilters2
        
        self.encoder = nn.Sequential(
            nn.Conv2d(ncolors, self.nfilters1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nfilters1, self.nfilters2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder_flatten = nn.Flatten()
        self.encoder_end = nn.Linear((patch_size//4)*(patch_size//4)*self.nfilters2 + ncolors, latent_dim + latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, (patch_size//4)*(patch_size//4)*self.nfilters2),
            nn.Unflatten(1, (self.nfilters2, patch_size//4, patch_size//4)),
            nn.ReLU(),
            nn.ConvTranspose2d(self.nfilters2, self.nfilters1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.nfilters1, ncolors, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, xs):
        x, sid_nums = xs
        output = self.encoder_flatten(self.encoder(x))
        avg_profile = x.mean(axis=(2,3))
        output = self.encoder_end(torch.cat((output, avg_profile), dim=1))
        mean, logvar = torch.split(output, self.latent_dim, dim=1)
        return mean, logvar

    def decode(self, zs):
        z, sid_nums = zs
        return self.decoder(z)

    def penultimate_layer(self, x : Tensor):
        return self.encoder(x)
