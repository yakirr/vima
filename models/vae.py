import torch
from torch import nn, Tensor
from abc import ABC, abstractmethod

class VAE(nn.Module, ABC):
    """Generic variational autoencoder."""

    def __init__(self):
        super().__init__()

    def reparameterize(self, mean : Tensor, logvar : Tensor):
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * .5) + mean

    def forward(self, x, sample_from_latent=True):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar) if sample_from_latent else mean
        x = self.decode(z)

        return x, mean, logvar

    @abstractmethod
    def encode(self, x : Tensor):
        pass #should output mean, logvar

    @abstractmethod
    def decode(self, x : Tensor):
        pass

    @abstractmethod
    def penultimate_layer(self, x : Tensor):
        pass #can output arbitary tensor with same number of observations as x

    # returns a flattened vector per observation without the variational jitter added
    # during training
    def embedding(self, x):
        return self.encode(x)[0].reshape((len(x), -1))