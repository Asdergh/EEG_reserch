import torch as th
from layers import *
from math import log2

class ImageEncoder(Module):

    def __init__(
        self,
        img_size: tuple[int],
        in_channels: int,
        out_channels: int,
        encoding_dim: int,
        hiden_channels: int = 32,
        patch_size: tuple[int] = (32, 32)
    ) -> None:

        super().__init__()
        self._net = Sequential(
            Conv2dSS(in_channels=in_channels, out_channels=hiden_channels),
            *[
                Conv2dSS(in_channels=hiden_channels, out_channels=hiden_channels) 
                for _ in range((int(log2(img_size[0])) - int(log2(patch_size[0]))) - 1)
            ],
            Conv2dSS(
                in_channels=hiden_channels, 
                out_channels=out_channels,
                stride=1
            ),
            Flatten(),
            MLPLayer(
                in_features=(patch_size[0] * patch_size[1] * out_channels), 
                out_features=encoding_dim
            )
        )
       
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)


class ImageDecoder(Module):

    def __init__(
        self,
        encoding_dim: int,
        img_size: tuple[int],
        in_channels: int,
        out_channels: int,
        hiden_channels: int = 32,
        patch_size: tuple[int] = (32, 32)
    ) -> None:
        
        super().__init__()
        self.patch_s = patch_size
        self.in_ch = in_channels
        self._projection = Linear(in_features=encoding_dim, out_features=(
            patch_size[0] * 
            patch_size[1] * 
            in_channels
        ))
        
        self._conv = Sequential(*[
            Conv2dTransposeSS(in_channels=in_channels, out_channels=hiden_channels)
            for _ in range((int(log2(img_size[0])) - int(log2(patch_size[0]))) - 1)
        ], Conv2dTransposeSS(in_channels=hiden_channels, out_channels=out_channels))


    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        proj = self._projection(inputs)
        proj = proj.view(size=(
            proj.size()[0],
            self.in_ch,
            self.patch_s[0],
            self.patch_s[1]
        ))
        return self._conv(proj)


class Img2ImgNet(Module):

    def __init__(
        self,
        img_size: tuple[int],
        in_channels: int,
        out_channels: int,
        encoding_dim: int,
        hiden_channels: int = 32,
        patch_size: tuple[int] = (32, 32),

    ) -> None:

        super().__init__()
        self._encoder = ImageEncoder(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            encoding_dim=encoding_dim,
            hiden_channels=hiden_channels,
            patch_size=patch_size
        )
        self._decoder = ImageDecoder(
            encoding_dim=encoding_dim,
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            hiden_channels=hiden_channels,
            patch_size=patch_size
        )

    def encode(self, inputs: th.Tensor) -> th.Tensor:
        return self._encoder(inputs)

    def decode(self, inputs: th.Tensor) -> th.Tensor:
        return self._decoder(inputs)
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        
        encoding = self._encoder(inputs)
        decoding = self._decoder(encoding)
        print(f"encoding size: [{encoding.size()}] \n decoding size: [{decoding.size()}]")
        return decoding



if __name__ == "__main__":

    model = Img2ImgNet(
        img_size=(512, 512),
        in_channels=3,
        out_channels=3,
        encoding_dim=32,
        patch_size=(128, 128)
    )
    A = th.normal(0.12, 1.12, (10, 3, 512, 512))
    print(model(A).size())