import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    UpSampling2D,
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    MaxPool2D,
    Concatenate,
    Add,
    Multiply,
    Flatten,
    LayerNormalization
)
from tensorflow.keras.losses import (
    MeanSquaredError,
    BinaryCrossentropy
)
from tensorflow.keras import (
    Model, 
    Sequential
)
from tensorflow import (
    Module,
    function
)
from tensorflow.keras.optimizers import Adam


class Conv(Module):

    def __init__(self, filters, kernel_size=3, strides=1, activation="tanh"):

        self.model_ = Sequential([
            Conv2D(
                filters=filters, 
                kernel_size=kernel_size,
                strides=strides,
                padding="same"
            ),
            BatchNormalization(),
            Activation(activation)
        ])
    
    def call(self, input):
        return self.model_(input)

class UpSample(Module):

    def __init__(self, filters, kernel_size=3, activation="tanh"):

        self.model_ = Sequential([
            Conv2D(
                filters=filters, 
                kernel_size=kernel_size,
                strides=1,
                padding="same"
            ),
            UpSample(size=2),
            BatchNormalization(),
            Activation(activation)
        ])
    
    def call(self, input):
        return self.model_(input)

class DownSample(Module):

    def __init__(self, filters, kernel_size=3, activation="tanh"):

        self.model_ = Sequential([
            Conv2D(
                filters=filters, 
                kernel_size=kernel_size,
                strides=1,
                padding="same"
            ),
            MaxPool2D(pool_size=2),
            BatchNormalization(),
            Activation(activation)
        ])
    
    def call(self, input):
        return self.model_(input)

class ResBlock(Module):

    def __init__(self, filters, kernel_size=3):

        self.res_conv_ = Sequential([
            Conv2D(
                filters=filters, 
                kernel_size=kernel_size,
                strides=1,
                padding="same"
            ),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(
                filters=filters, 
                kernel_size=kernel_size,
                strides=1,
                padding="same"
            ),
            BatchNormalization()
        ])
        self.conv_ = Conv2D(
                filters=filters, 
                kernel_size=1,
                strides=1,
                padding="same"
            )
    
    def call(self, input):

        x = self.res_conv_(input)
        conv = self.conv_(input)

        x = Concatenate([x, conv])
        x = Activation("relu")(x)
        return x

class ResNet(Module):

    def __init__(self):

        self.model_ = Sequential([
            ResBlock(filtesr=128),
            ResBlock(filtesr=64),
            ResBlock(filtesr=3),
        ])
    
    def call(self, input):
        return self.model_(input)

class Unet(Module):

    def __init__(self, input_channels=3):

        self.input_ch = input_channels
        self.down0 = DownSample(filters=128)
        self.down1 = DownSample(filters=64)
        self.down2 = DownSample(filters=32)

        self.up0 = UpSample(filters=32)
        self.up1 = UpSample(filters=64)
        self.up2 = UpSample(filters=128)
    
    def call(self, input):

        down0 = self.down0(input)
        down1 = self.down1(down0)
        down2 = self.down2(down1)

        up0 = self.up0(down2)
        up1 = self.up1(up0)
        up2 = self.up2(up1)

        out = Conv2D(filters=self.input_ch)(up2)
        out = BatchNormalization()(out)
        return out
    


class DenseNet(Module):

    def __init__(self, input_channels=3, dp_rate=0.45):

        self.input_ch = input_channels
        self.model_ = Sequential([
            Flatten(),
            LayerNormalization(),
            Dense(units=128, activation="relu"),
            Dropout(rate=dp_rate),
            Dense(units=64, activation="relu"),
            Dropout(rate=dp_rate),
            Dense(units=32, activation="relu"),
            Dropout(rate=dp_rate),
            Dense(units=1, activation="sigmoid"),

        ])