from torch import nn, Tensor
import torch

from .vae import VAE
from .resnet_using_basic_block_encoder import Encoder, BasicBlockEnc
from .resnet_using_basic_block_decoder import Decoder, BasicBlockDec
from .resnet_using_light_basic_block_encoder import LightEncoder, LightBasicBlockEnc
from .resnet_using_light_basic_block_decoder import LightDecoder, LightBasicBlockDec

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

    def __init__(self, network='default', num_layers=18, nlatent=100, ncolors=5, variational=True):
        """Initialize the autoencoder.

        Args:
            network (str): a flag to efine the network version. Choices ['default' (default), 'light'].
             num_layers (int): the number of layers to be created. Choices [18 (default), 34 (only for 
                'default' network), 20 (only for 'light' network).
        """
        super().__init__()
        self.variational = variational
        self.network = network
        if self.network == 'default':
            if num_layers==18:
                self.encoder = Encoder(BasicBlockEnc, [2, 2, 2, 2])
                self.decoder = Decoder(BasicBlockDec, [2, 2, 2, 2])
            elif num_layers==34:
                self.encoder = Encoder(BasicBlockEnc, [3, 4, 6, 3])
                self.decoder = Decoder(BasicBlockDec, [3, 4, 6, 3]) 
            else:
                raise NotImplementedError("Only resnet 18 & 34 autoencoder have been implemented for images size >= 64x64.")
        elif self.network == 'light':
            if num_layers==18:
                self.encoder = LightEncoder(ncolors, nlatent, LightBasicBlockEnc, [2, 2, 2]) 
                self.decoder = LightDecoder(ncolors, nlatent, LightBasicBlockDec, [2, 2, 2]) 
            elif num_layers==20:
                self.encoder = LightEncoder(ncolors, nlatent, LightBasicBlockEnc, [3, 3, 3]) 
                self.decoder = LightDecoder(ncolors, nlatent, LightBasicBlockDec, [3, 3, 3]) 
            else:
                raise NotImplementedError("Only resnet 18 & 20 autoencoder have been implemented for images size < 64x64.")
        else:
                raise NotImplementedError("Only default and light resnet have been implemented. The light version corresponds to input datasets with size less than 64x64.")

    def encode(self, x : Tensor):
        return self.encoder(x)

    def decode(self, z : Tensor):
        return self.decoder(z)

    def penultimate_layer(self, x : Tensor):
        return self.encoder(x)[0]