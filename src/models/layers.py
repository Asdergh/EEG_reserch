import torch as th
import numpy as np

from torch.nn import (
    Conv2d,
    Conv1d,
    Linear,
    Softmax,
    Sequential,
    Module,
    ModuleDict,
    BatchNorm2d,
    BatchNorm1d,
    Tanh,
    Upsample,
    ConvTranspose2d,
    Sigmoid,
    Softmax,
    ReLU,
    Flatten,
    LayerNorm,
    Dropout,
    BatchNorm3d,
    Conv3d,
    ConvTranspose3d
)


__activations__ = {
    "tanh": Tanh,
    "sigmoid": Sigmoid,
    "softmax": Softmax,
    "relu": ReLU,

}
__all__ = [
    "Flatten",
    "Conv2dSS",
    "Conv1dSS",
    "UpsampleLayer",
    "ResLayer",
    "MulHeadAttention",
    "Module",
    "ModuleDict",
    "BatchNorm2d",
    "BatchNorm1d",
    "Tanh",
    "Upsample",
    "Conv2d",
    "Conv1d",
    "Linear",
    "Softmax",
    "Sequential",
    "Conv2dTransposeSS",
    "MLPLayer",
    "LayerNorm",
    "Dropout",
    "ReLU",
    "Conv3dSS",
    "Conv3dTransposeSS"
]



class MLPLayer(Module):
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hiden_features: int = 32,
        hiden_l: int = 3,
        activation: str = "relu",
        out_activation: str = "relu"
    ) -> None:
        
        super().__init__()
        self._net = Sequential(
            Linear(in_features=in_features, out_features=hiden_features),
            *[Sequential(
                Linear(in_features=hiden_features, out_features=hiden_features),
                LayerNorm(normalized_shape=hiden_features),
                __activations__[activation]()
            ) for _ in range(hiden_l)],
            Linear(in_features=hiden_features, out_features=out_features),
            __activations__[out_activation]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)


class Conv3dSS(Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 1,
        stride: int = 2,
        kernel_size: int = 3,
        activation: str = "relu"
    ) -> None:
        
        super().__init__()
        self._net = Sequential(
            Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                kernel_size=kernel_size
            ),
            BatchNorm3d(num_features=out_channels),
            __activations__[activation]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)

class Conv3dTransposeSS(Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        padding: int = 1,
        stride: int = 2,
        kernel_size: int = 4,
        activation: str = "relu"
    ) -> None:
        
        super().__init__()
        self._net = Sequential(
            ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                padding=padding,
                stride=stride,
                kernel_size=kernel_size
            ),
            BatchNorm3d(num_features=out_channels),
            __activations__[activation]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)

    

class Conv2dTransposeSS(Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        padding: int = 1,
        stride: int = 2,
        kernel_size: int = 4,
        activation: str = "relu"
    ) -> None:
        
        super().__init__()
        self._net = Sequential(
            ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                padding=padding,
                stride=stride,
                kernel_size=kernel_size
            ),
            BatchNorm2d(num_features=out_channels),
            __activations__[activation]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)
    
class Conv2dSS(Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 1,
        stride: int = 2,
        kernel_size: int = 3,
        activation: str = "tanh"
    ) -> None:
        
        super().__init__()
        self._net = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                kernel_size=kernel_size
            ),
            BatchNorm2d(num_features=out_channels),
            __activations__[activation]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)
    
    
class Conv1dSS(Module):
    
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        padding: int = 1,
        stride: int = 2,
        kernel_size: int = 3,
        activation: str = "tanh"
    ) -> None:

        super().__init__()
        self._net = Sequential(
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                kernel_sie=kernel_size
            ),
            BatchNorm1d(num_features=out_channels),
            __activations__[activation]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)

class UpsampleLayer(Module):

    def __init__(
        self, 
        in_channels: int, 
        activation: str = "tanh"
    ) -> None:
        
        super().__init__()
        self._net = Sequential(
            Upsample(scale_factor=2),
            BatchNorm2d(num_features=in_channels),
            __activations__[activation]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)

class ResLayer(Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        
        super().__init__()
        self._projection = Linear(in_features=in_features, out_features=out_features)
    
    def __call__(self, inputs: list[th.Tensor]) -> th.Tensor:
        return th.add(self._projection(inputs[0]), inputs[1])
    
class MulHeadAttention(Module):

    def __init__(
        self,
        in_features: int,
        out_features: int
    ) -> None:
        
        super().__init__()
        self.d = th.tensor(out_features)
        self._projections = ModuleDict({
            "q": Linear(in_features=in_features, out_features=out_features),
            "k": Linear(in_features=in_features, out_features=out_features),
            "v": Linear(in_features=in_features, out_features=out_features)
        })
        self._act = Softmax(dim=1)
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        return th.mm(
            self._act(th.mm(self._projections["q"](inputs), self._projections["k"](inputs).T)),
            self._projections["v"](inputs)
        )