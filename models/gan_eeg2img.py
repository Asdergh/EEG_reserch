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
    LayerNormalization,
    Conv1D,
    Reshape
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
    function,
    GradientTape
)
from tensorflow.keras.metrics import Mean
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


    

class MelEmbedder(Module):

    def __init__(self):
        
        self.model_ = Sequential([
            Conv1D(
                filters=128,
                kernel_size=3,
                strides=1,
                padding="same"
            ),
            BatchNormalization(),
            Conv1D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding="same"
            ),
            BatchNormalization(),
            Flatten(),
            Dense(units=128, activation="relu"),
            Dense(units=64, activation="relu"),
            Dense(units=1, activation="relu")
        ])
    
    def call(self, input):
        return self.model_(input)


class UpNet(Module):

    def __init__(self, out_channels=3):

        self.out_channels = out_channels
        self.model_ = Sequential([
            UpSample(filters=128),
            UpSample(filters=64),
            UpSample(filters=32),
            Conv2D(
                filters=out_channels,
                kernel_size=3,
                strides=1,
                padding="same"
            ),
            Activation("tanh")
        ])
    
    def all(self, input):
        return self.model_(input)
    

class DenseNet(Module):

    def __init__(self, input_channels=3, dp_rate=0.45):

        self.input_ch = input_channels
        self.model_ = Sequential([
            Flatten(),
            LayerNormalization(),
            Dense(units=128, activation="relu"),
            Dropout(rate=dp_rate),
            LayerNormalization(),
            Dense(units=64, activation="relu"),
            Dropout(rate=dp_rate),
            LayerNormalization(),
            Dense(units=32, activation="relu"),
            Dropout(rate=dp_rate),
            Dense(units=1, activation="sigmoid")
        ])
    
    def call(self, input):
        return self.model_(input)


class GanModel(Model):

    def __init__(self, input_sh, emb_dim,  mel_sh, **kwargs):

        super(self, **kwargs).__init__()
        self.input_sh = input_sh
        self.emd_dim = emb_dim
        self.mel_sh = mel_sh
        self.gen_, self.dis_ = self.build_models_()

    
    def build_models_(self):

        dis_input = Input(shape=self.input_sh)
        enc_input = Input(shpe=self.mel_sh)

        encoder = MelEmbedder()(enc_input)
        rec_sh = Dense(units=(self.input_sh[-1] * (self.input_sh[0] // 4) * (self.input_sh[1] // 4)))(encoder)
        rec_sh = Reshape(target_shape=(
            self.input_sh[0] // 4,
            self.input_sh[1] // 4,
            self.input_sh[2]
        ))(rec_sh)

        upsampling = UpNet(out_channels=3)(rec_sh)
        dense_net = DenseNet()(dis_input)

        return (
            Model(inputs=enc_input, outputs=upsampling),
            Model(inputs=dis_input, outputs=dense_net)
        )
        

    def compile(self, optimizers, losses):

        super().compile()
        self.gen_optimizer, self.dis_optimizer = optimizers
        self.gen_lfn, self.dis_lfn = losses 
        
        self.gen_loss_tracker = Mean(name="gan_loss_tracker")
        self.dis_loss_tracker = Mean(name="dis_loss_tracker")
        self.total_loss_tracker = Mean(name="dsi_loss_tracker") 
    
    def trainig_step(self, inputs):

        
        mels_batch, images_batch = inputs
        dis_valid_scores = np.ones(mels_batch.shape[0])
        dis_fake_scores = np.zeros(mels_batch.shape[0])
        with GradientTape() as gen_tape, GradientTape() as dis_tape:

            gen_out = self.gen_(mels_batch)
            dis_valid_out, dis_fake_out = self.dis_(images_batch), self.dis_(gen_out)
            
            dis_loss = self.dis_lfn(dis_valid_out, dis_valid_scores) + self.dis_fn(dis_fake_out, dis_fake_scores)
            gen_loss = self.gen_lfn(images_batch, gen_out)
            total_loss = dis_loss + gen_loss
        
        gen_grads = gen_tape.gradient(total_loss, self.gen_.trainable_variables)
        dis_grads = dis_tape.gradient(total_loss, self.dis_.trainables_variables)
        
        self.gen_optimizer.apply_gradients(zip(total_loss, gen_grads))
        self.dis_optimizer.apply_gradients(zip(total_loss, dis_grads))

        self.gen_loss_tracker.update_state(gen_loss)
        self.dis_loss_tracker.update_state(dis_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            self.gen_loss_tracker.name: self.gen_loss_tracker.result(),
            self.dis_loss_tracker.name: self.dis_loss_tracker.result(),
            self.total_loss_tracker.name: self.total_loss_tracker.result()
        }
            

    
