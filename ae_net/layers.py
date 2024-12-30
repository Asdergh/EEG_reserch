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
    Upsample
)

class Conv2dSS(Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 0,
        stride: int = 2,
        kernel_size: int = 3
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
            Tanh()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)
    
    
class Conv1dSS(Module):
    
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        padding: int = 0,
        stride: int = 2,
        kernel_size: int = 3
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
            Tanh()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)

class UpsampleLayer(Module):

    def __init__(self, in_channels: int) -> None:
        
        super().__init__()
        self._net = Sequential(
            Upsample(scale_factor=2),
            BatchNorm2d(num_features=in_channels),
            Tanh()
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