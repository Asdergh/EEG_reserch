import numpy as np
import torch as th
from layers import *
from math import log2


class Encoder3D(Module):

    def __init__(
        self,
        in_size: tuple[int],
        att_features: int,
        encoding_dim: int
    ):
        super().__init__()
        self._conv_ = Conv3dSS(
            in_channels=1,
            out_channels=1
        )
        self._flt_ = Flatten()
        self._lin_net_ = Sequential(
            MulHeadAttention(
                in_features=(
                (in_size[0] // 2)
                * (in_size[1] // 2)
                * (in_size[2] // 2)),
                out_features=att_features
            ),
            MLPLayer(
                in_features=att_features,
                out_features=encoding_dim
            )
        )
        self._layers_list_ = [
            self._conv_,
            self._flt_,
            self._lin_net_
        ]
    
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        
        conv = self._conv_(inputs)
        flatten = self._flt_(conv)
        print(flatten.size())
        return self._lin_net_(flatten)
    
    def get_activation_maps(self, inputs: th.Tensor) -> list[th.Tensor]:

        outs = []
        x = inputs
        for layer in self._layers_:
            x = layer(x)
            outs.append(x)
        
        return outs


class Decoder3D(Module):

    def __init__(
        self,
        encoding_dim: int,
        out_size: tuple[int],
        patch_size: tuple[int] = (32, 32, 32)
    ):
        super().__init__()
        conv_n = int(log2(out_size[0])) - int(log2(patch_size[0]))
        self.patch_s = patch_size
        self._projection_ = Linear(in_features=encoding_dim, out_features=(
            patch_size[0] * 
            patch_size[1] * 
            patch_size[2]
        ))
        self._de_conv_ = ModuleDict({
        f"conv_{i}": Conv3dTransposeSS(in_channels=1, out_channels=1)
            for i in range(conv_n)
        })
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        
        x = self._projection_(inputs)
        x = x.view((
            x.size()[0],
            1,
            self.patch_s[0],
            self.patch_s[1],
            self.patch_s[2]
        ))
        for key in self._de_conv_.keys():
            x = self._de_conv_[key](x)

        return x

    def get_activation_maps(self, inputs: th.Tensor) -> list[th.Tensor]:

        outs = []
        x = inputs
        for key in self._de_conv_.keys():
            x = self._de_conv_[key](x)
            outs.append(x)
        
        return outs


class VolAeNet(Module):

    def __init__(
        self,
        in_size: tuple[int],
        att_features: int,
        encoding_dim: int,
        patch_size: tuple[int] = (32, 32, 32)
    ) -> None:
        
        super().__init__()
        self._encoder_ =  Encoder3D(
            in_size=in_size,
            att_features=att_features,
            encoding_dim=encoding_dim
        )
        self._decoder_ = Decoder3D(
            encoding_dim=encoding_dim,
            out_size=in_size,
            patch_size=patch_size
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        
        encoding = self._encoder_(inputs)
        return self._decoder_(encoding)

if __name__ == "__main__":

    A = th.normal(0.12, 1.12, (2, 1, 128, 128, 128))
    encoder = Encoder3D(
        in_size=(128, 128, 128),
        att_features=128,
        encoding_dim=32
    )
    decoder = Decoder3D(
        encoding_dim=32,
        out_size=(128, 128, 128),
        patch_size=(64, 64, 64)
    )

    print(f"Encoder test: {encoder(A).size()}")
    print(f"Decoder test: {decoder(encoder(A)).size()}")

    
        


    
