from torch import nn, Tensor
import torch

from .vae import VAE

from . import resnetlight_simple_encoder as simple_enc
from . import resnetlight_simple_decoder as simple_dec
from . import resnetlight_advanced_encoder as adv_enc
from . import resnetlight_advanced_decoder as adv_dec

# based on https://github.com/eleannavali/resnet-18-autoencoder

class ResnetVAE(VAE):
    """Resnet-inspired VAE.

    Attributes:
        network (str): the architectural type of the network. There are 2 choices:
            - 'default' (default), related with the original resnet-18 architecture
            - 'light', a samller network implementation of resnet-18 for smaller input images.
        num_layers (int): the number of layers to be created. Implemented for 18 layers (default) for both types 
            of network, 34 layers for default only network and 20 layers for light network. 
    """

    def __init__(self, nmarkers, nsids, network='light', mode='advanced', num_layers=18, nlatent=100, variational=True):
        """Initialize the autoencoder.

        Args:
            network (str): a flag to efine the network version. Choices ['default' (default), 'light'].
             num_layers (int): the number of layers to be created. Choices [18 (default), 34 (only for 
                'default' network), 20 (only for 'light' network).
        """
        super().__init__()        
        self.variational = variational
        self.network = network
        self.mode = mode

        if self.network == 'default':
            # if num_layers==18:
            #     self.encoder = Encoder(BasicBlockEnc, [2, 2, 2, 2])
            #     self.decoder = Decoder(BasicBlockDec, [2, 2, 2, 2])
            # elif num_layers==34:
            #     self.encoder = Encoder(BasicBlockEnc, [3, 4, 6, 3])
            #     self.decoder = Decoder(BasicBlockDec, [3, 4, 6, 3]) 
            # else:
            #     raise NotImplementedError("Only resnet 18 & 34 autoencoder have been implemented for images size >= 64x64.")
            raise NotImplementedError("Only light network is supported currently.")
        elif self.network == 'light':
            enc = simple_enc if mode == 'simple' else adv_enc
            dec = simple_dec if mode == 'simple' else adv_dec
            if num_layers==18:
                self.encoder = enc.LightEncoder(nmarkers, nsids, nlatent, enc.LightBasicBlockEnc, [2, 2, 2]) 
                self.decoder = dec.LightDecoder(nmarkers, nsids, nlatent, dec.LightBasicBlockDec, [2, 2, 2]) 
            elif num_layers==20:
                self.encoder = enc.LightEncoder(nmarkers, nlatent, enc.LightBasicBlockEnc, [3, 3, 3]) 
                self.decoder = dec.LightDecoder(nmarkers, nlatent, dec.LightBasicBlockDec, [3, 3, 3]) 
            else:
                raise NotImplementedError("Only resnet 18 & 20 autoencoder have been implemented for images size < 64x64.")
        else:
                raise NotImplementedError("Only default and light resnet have been implemented. The light version corresponds to input datasets with size less than 64x64.")

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def penultimate_layer(self, x):
        return self.encoder(x)[0]