import torch as th
from math import log2
from layers import *



class GeneratorNet(Module):

    def __init__(
        self,
        noise_dim: int,
        out_channels: int,
        img_size: tuple[int],
        patch_size: tuple[int] = None,
        hiden_channels: int = 32
    ) -> None:
        
        super().__init__()
        conv_n = int(log2(img_size[0])) - int(log2(patch_size[0]))
        if patch_size is None:
            patch_size = (
                img_size[0] // 3,
                img_size[1] // 3
            )
        self.patch_size = patch_size
        
        self._proj = Linear(in_features=noise_dim, out_features=patch_size[0] * patch_size[1])
        self._conv = Sequential(
            Conv2dSS(
                in_channels=1,
                out_channels=hiden_channels,
                padding=1,
                stride=1
            ),
            *[
                Conv2dTransposeSS(in_channels=hiden_channels, out_channels=hiden_channels)
                for _ in range(conv_n)
            ],
            Conv2dSS(
                in_channels=hiden_channels,
                out_channels=out_channels,
                padding=1,
                stride=1
            )
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        
        proj = self._proj(inputs)
        proj = th.unsqueeze(proj.view(
            proj.size()[0],
            self.patch_size[0],
            self.patch_size[1],
        ), dim=1)
        return self._conv(proj)
    
class DiscriminatorNet(Module):

    def __init__(
        self,
        img_size: tuple[int],
        in_channels: int,
        att_features: int = 128,
        patch_size: tuple[int] = (32, 32)
    ) -> None:
        
        super().__init__()
        conv_s = int(log2(img_size[0])) - int(log2(patch_size[0]))
        self._conv = Sequential(*[
            Conv2dSS(in_channels=in_channels, out_channels=in_channels)
            for _ in range(conv_s)
        ])

        self._flt = Flatten()
        self._mlp_att = Sequential(
            MulHeadAttention(in_features=(
                img_size[0] // (conv_s * 2) *
                img_size[1] // (conv_s * 2) * 
                in_channels
            ), out_features=att_features),
            LayerNorm(normalized_shape=att_features),
            Dropout(p=0.45),
            MLPLayer(
                in_features=att_features,
                out_features=1,
                out_activation="sigmoid"
            )
        )
        

    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        
        conv = self._conv(inputs)
        flatten = self._flt(conv)
        return self._mlp_att(flatten)
    
if __name__ == "__main__":
    
    generator = GeneratorNet(
        img_size=(128, 2128),
        patch_size=(64, 64),
        noise_dim=32,
        hiden_channels=128,
        out_channels=3
    )
    discriminator = DiscriminatorNet(
        img_size=(128, 128),
        in_channels=3
    )
    A = th.normal(0.12, 1.12, (32, 32))
    print(generator(A).size())
    print(discriminator(generator(A)).size())
    





        
        
        