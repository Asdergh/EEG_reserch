import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

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

        super(Conv, self).__init__()
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
    
    def __call__(self, input):
        return self.model_(input)

class UpSample(Module):

    def __init__(self, filters, kernel_size=3, activation="tanh"):

        super(UpSample, self).__init__()
        self.model_ = Sequential([
            Conv2D(
                filters=filters, 
                kernel_size=kernel_size,
                strides=1,
                padding="same"
            ),
            UpSampling2D(size=2),
            BatchNormalization(),
            Activation(activation)
        ])
    
    def __call__(self, input):
        return self.model_(input)

class DownSample(Module):

    def __init__(self, filters, kernel_size=3, activation="tanh"):

        super(DownSample, self).__init__()
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
    
    def __call__(self, input):
        return self.model_(input)

class ResBlock(Module):

    def __init__(self, filters, kernel_size=3):

        super(ResBlock, self).__init__()
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
    
    def __call__(self, input):

        x = self.res_conv_(input)
        conv = self.conv_(input)

        x = Concatenate([x, conv])
        x = Activation("relu")(x)
        return x


    

class MelEmbedder(Module):

    def __init__(self):
        
        super(MelEmbedder, self).__init__()
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
    
    def __call__(self, input):
        return self.model_(input)


class UpNet(Module):

    def __init__(self, out_channels=3):

        super(UpNet, self).__init__()
        self.out_channels = out_channels
        self.model_ = Sequential([
            UpSample(filters=128),
            UpSample(filters=64),
            Conv2D(
                filters=out_channels,
                kernel_size=3,
                strides=1,
                padding="same"
            ),
            Activation("tanh")
        ])
    
    def __call__(self, input):
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
    
    def __call__(self, input):
        return self.model_(input)


class GanModel(Model):

    def __init__(self, input_sh, mel_sh, **kwargs):

        super().__init__(**kwargs)
        self.input_sh = input_sh
        self.mel_sh = mel_sh
        self.gen_, self.dis_ = self.build_models_()

    
    def build_models_(self):

        dis_input = Input(shape=self.input_sh)
        enc_input = Input(shape=self.mel_sh)

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
        

    def compile(self, optimizer, loss):

        super().compile()
        self.gen_optimizer, self.dis_optimizer = optimizer
        self.gen_lfn, self.dis_lfn = loss
        
        self.gen_loss_tracker = Mean(name="gan_loss_tracker")
        self.dis_loss_tracker = Mean(name="dis_loss_tracker")
        self.total_loss_tracker = Mean(name="total_loss_tracker") 
    
    def train_step(self, inputs):

        
        mels_batch, images_batch = inputs
        dis_valid_scores = np.ones(mels_batch.shape[0])
        dis_fake_scores = np.zeros(mels_batch.shape[0])
        with GradientTape() as gen_tape, GradientTape() as dis_tape:

            gen_out = self.gen_(mels_batch)
            dis_valid_out, dis_fake_out = self.dis_(images_batch), self.dis_(gen_out)
            
            dis_loss = self.dis_lfn(dis_valid_out, dis_valid_scores) + self.dis_lfn(dis_fake_out, dis_fake_scores)
            gen_loss = self.gen_lfn(images_batch, gen_out)
            total_loss = tf.cast(dis_loss, tf.float64) + tf.cast(gen_loss, tf.float64)
        
        gen_grads = gen_tape.gradient(total_loss, self.gen_.trainable_variables)
        dis_grads = dis_tape.gradient(total_loss, self.dis_.trainable_variables)
        
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.gen_.trainable_variables))
        self.dis_optimizer.apply_gradients(zip(dis_grads, self.dis_.trainable_variables))

        self.gen_loss_tracker.update_state(gen_loss)
        self.dis_loss_tracker.update_state(dis_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            self.gen_loss_tracker.name: self.gen_loss_tracker.result(),
            self.dis_loss_tracker.name: self.dis_loss_tracker.result(),
            self.total_loss_tracker.name: self.total_loss_tracker.result()
        }
        
    function
    def call(self, inputs):
        return self.gen_(inputs)

    
if __name__ == "__main__":


    gan_net = GanModel(input_sh=(128, 128, 3), mel_sh=(100, 45))
    gan_net.gen_.summary()
    gan_net.dis_.summary()

    random_mels = np.random.normal(0, 12.0, (200, 100, 45))
    random_images = np.random.normal(0, 12.0, (200, 128, 128, 3))
    
    # gen_out = gan_net.gen_.predict(random_mels)
    # dis_out = gan_net.dis_.predict(random_images)
    
    # gan_net.compile(
    #     optimizer=[
    #         Adam(learning_rate=0.01),
    #         Adam(learning_rate=0.01)
    #     ],
    #     loss=[
    #         MeanSquaredError(),
    #         BinaryCrossentropy()
    #     ]
    # )
    # gan_net.fit(
    #     random_mels, random_images,
    #     epochs=10, 
    #     batch_size=32
    # )

    # gan_net.save_weights("C:\\Users\\1\\Desktop\\EegProject\\models\\gan_weights.weights.h5")
    gan_net.load_weights("C:\\Users\\1\\Desktop\\EegProject\\models\\gan_weights.weights.h5")
    gen_out = gan_net.gen_.predict(random_mels)
    dis_out = gan_net.dis_.predict(random_images)
    print(gen_out.shape, dis_out.shape)

    