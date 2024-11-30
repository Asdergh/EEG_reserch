import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os


plt.style.use("dark_background")
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
from tensorflow.keras.callbacks import Callback


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

    def __init__(self, embedding_dim: int) -> None:
        
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
            Dense(units=embedding_dim, activation="relu")
        ])
    
    def __call__(self, input):
        return self.model_(input)


class UpNet(Module):

    def __init__(self, out_channels=3):

        super(UpNet, self).__init__()
        self.out_channels = out_channels
        self.model_ = Sequential([
            UpSample(filters=128, activation="linear"),
            UpSample(filters=64, activation="linear"),
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



class GanModelCallback(Callback):

    def __init__(
            self, 
            gen_model, 
            data, 
            run_folder,
            img_sh=(128, 128, 3),
            samples_n=25,
    ):

        
        self.gen_model = gen_model
        self.data = data
        self.img_sh = img_sh
        self.samples_n = samples_n
        self.run_folder = run_folder
    
    def on_epoch_end(self, epoch, logs=None):

        fig, axis = plt.subplots()
        show_tensor = np.zeros((
            int(np.sqrt(self.samples_n)) * self.img_sh[0],
            int(np.sqrt(self.samples_n)) * self.img_sh[1],
            self.img_sh[-1]
        ))

        for i in range(int(np.sqrt(self.samples_n))):
            for j in range(int(np.sqrt(self.samples_n))):

                
                random_idx = np.random.randint(0, self.data.shape[0])
                mel_sample = self.data[random_idx]
                mel_sample = np.expand_dims(mel_sample, axis=0)
                print(mel_sample.shape, self.data.shape)
                generated_img = self.gen_model.predict(mel_sample, verbose=0)
                show_tensor[
                    i * self.img_sh[0]: (i + 1) * self.img_sh[0],
                    j * self.img_sh[1]: (j + 1) * self.img_sh[1],
                    :
                ] = generated_img
                
            
        gen_path = os.path.join(self.run_folder, "gen_samples")
        if not os.path.exists(gen_path):
            os.mkdir(gen_path)
        
        epoch_path = os.path.join(gen_path, f"generation_at_{epoch}.png")
        axis.imshow(show_tensor)
        fig.savefig(epoch_path)
    
        
        
        
class GanModel(Model):

    def __init__(
            self, 
            input_sh: tuple, 
            mel_sh: tuple, 
            embedding_dim: int, 
            **kwargs
    ) -> None:

        super().__init__(**kwargs)
        self.input_sh = input_sh
        self.mel_sh = mel_sh
        self.emb_dim = embedding_dim
        self.gen_, self.dis_, self.enc_ = self.build_models_()


    def build_models_(self):

        dis_input = Input(shape=self.input_sh)
        enc_input = Input(shape=self.mel_sh)

        encoder = MelEmbedder(embedding_dim=self.emb_dim)(enc_input)
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
            Model(inputs=dis_input, outputs=dense_net),
            Model(inputs=enc_input, outputs=encoder)
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
            gen_loss = self.gen_lfn(tf.ones_like(dis_fake_out), dis_fake_out)
            total_loss = tf.cast(dis_loss, tf.float64) + tf.cast(gen_loss, tf.float64)
        
        gen_grads = gen_tape.gradient(gen_loss, self.gen_.trainable_variables)
        dis_grads = dis_tape.gradient(dis_loss, self.dis_.trainable_variables)
        
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

    




# if __name__ == "__main__":

#     IMAGES_SH = (128, 128, 3)
#     MELS_SIZE = (14, 32)
#     train_mels = np.random.normal(0, 1.0, (100, 14, 32))
#     train_images = np.random.normal(0, 1.0, (100, 128, 128, 3))

#     model = GanModel(
#     input_sh=IMAGES_SH,
#     mel_sh=MELS_SIZE
#     )

#     callback = GanModelCallback(
#     run_folder="C:\\Users\\1\\Desktop\\EegProject\\models_storage\\gan_logs",
#     gen_model=model.gen_,
#     data=train_mels
#     )

#     model.compile(
#         optimizer=[
#             Adam(learning_rate=0.001),
#             Adam(learning_rate=0.001)
#         ],
#         loss=[
#             BinaryCrossentropy(),
#             BinaryCrossentropy()
#             ]
#     )

#     model_hs = model.fit(
#         train_mels, train_images,
#         epochs=100,
#         batch_size=32,
#         callbacks=[callback]
#     )