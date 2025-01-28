import torch as th
from layers import *
from mels2img_net import TransformerEncoder
from vol_cnn import Decoder3D



class Eeg2VolumeNet(Module):

    def __init__(
        self,
        mels_size: tuple[int],
        encoding_dim: int,
        out_size: tuple[int],
        patch_size: tuple[int] = (32, 32, 32),
        att_features: int = 128,
        hiden_features: int = 64
        
    ) -> None:
        
        super().__init__()
        self._net_ = Sequential(
            TransformerEncoder(
                mels_size,
                encoding_dim,
                att_features,
                hiden_features
            ),
            Decoder3D(
                encoding_dim,
                out_size,
                patch_size
            )
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net_(inputs)
    


