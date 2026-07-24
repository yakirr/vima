import inspect
import torch
from torch import nn, Tensor
from abc import ABC, abstractmethod

class VAE(nn.Module, ABC):
    """Generic variational autoencoder."""

    def __init__(self):
        super().__init__()
        self.variational = True

    def reparameterize(self, mean : Tensor, logvar : Tensor):
        """Sample from the latent Gaussian using the reparameterization trick."""
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * .5) + mean

    def forward(self, xs, sample_from_latent=True):
        """
        Encode, sample a latent, and decode a batch of patches.

        Parameters
        ----------
        sample_from_latent
            Draw the latent via reparameterization; if false (or the model is
            non-variational), use the posterior mean directly.

        Returns
        -------
        tuple
            The reconstruction, latent mean, and latent log-variance.
        """
        _, sid_nums = xs
        mean, logvar = self.encode(xs)
        z = self.reparameterize(mean, logvar) if sample_from_latent and self.variational else mean
        x = self.decode((z, sid_nums))

        return x, mean, logvar

    @abstractmethod
    def encode(self, x : Tensor):
        """Map a batch of patches to the latent mean and log-variance."""
        pass #should output mean, logvar

    @abstractmethod
    def decode(self, x : Tensor):
        """Reconstruct patches from latent codes."""
        pass

    @abstractmethod
    def penultimate_layer(self, x : Tensor):
        """Return the features feeding the latent projection."""
        pass #can output arbitary tensor with same number of observations as x

    # returns a flattened vector per observation without the variational jitter added
    # during training
    def embedding(self, x):
        """Return the flattened latent mean per observation, without variational jitter."""
        return self.encode(x)[0].reshape((len(x[0]), -1))