import torch as th
from math import log2
from torch.nn import (
    Linear,
    Flatten,
    Sequential,
    Module,
)
from layers import *

class TransformerEncoder(Module):

    def __init__(
        self,
        mels_size: tuple[int],
        out_features: int,
        att_features: int = 128,
        hiden_features: int = 64
    ) -> None:

        super().__init__()
        self._conv = Sequential(
            Conv2dSS(in_channels=1, out_channels=32),
            Conv2dSS(in_channels=32, out_channels=64),
            Conv2dSS(in_channels=64, out_channels=32),
        )   
        self._flt = Flatten()
        self._att = MulHeadAttention(in_features=(mels_size[0] * mels_size[1]), out_features=att_features)
        self._res0 = ResLayer(in_features=(mels_size[0] * mels_size[1]), out_features=att_features)

        self._linear = Sequential(
            Linear(in_features=att_features, out_features=hiden_features),
            Linear(in_features=hiden_features, out_features=hiden_features),
            Linear(in_features=hiden_features, out_features=out_features)
        )
        self._res1 = ResLayer(in_features=att_features, out_features=out_features)
        
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        inputs = th.unsqueeze(inputs, dim=1)
        flt = self._flt(inputs)
        att = self._att(flt)
        return self._linear(att)
    
class ImgDecoder(Module):

    def __init__(
        self,
        in_features: int,
        out_channels: int,
        patch_size: int = 32,
        out_size: int = 512
    ) -> None:
        
        super().__init__()
        self.patch_s = patch_size
        self._proj = Linear(in_features=in_features, out_features=(patch_size ** 2))
        self._net = Sequential(
            Conv2dSS(
                in_channels=1,
                out_channels=out_channels,
                stride=1,
                padding=1
            ),
            *[
                UpsampleLayer(in_channels=out_channels)
                for _ in range((int(log2(out_size)) - int(log2(patch_size))))
            ]
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        proj = self._proj(inputs)
        proj = th.unsqueeze(proj.view((proj.size()[0], self.patch_s, self.patch_s)), dim=1)
        return self._net(proj)



class AeEegNet(Module):

    def __init__(
        self,
        mels_size: tuple[int],
        encoder_out_features: int,
        out_channels: int,
        patch_size: int = 32,
        out_size: int = 512,
        att_features: int = 128,
        hiden_features: int = 64
        
    ) -> None:
        
        super().__init__()
        self._net = ModuleDict({
            "encoder": TransformerEncoder(
                mels_size=mels_size,
                out_features=encoder_out_features,
                att_features=att_features,
                hiden_features=hiden_features
            ),
            "decoder": ImgDecoder(
                in_features=encoder_out_features,
                out_channels=out_channels,
                patch_size=patch_size,
                out_size=out_size
            )
        })
            
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        encoding = self._net["encoder"](inputs)
        return self._net["decoder"](encoding)

