import numpy as np
import os

from tensorflow.keras.layers import (
    Input, 
    Dense,
    Conv1D,
    AvgPool1D,
    Flatten,
    Activation,
    BatchNormalization,
    Dropout
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
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam



class Conv(Module):

    def __init__(self, filters, kernel_size, activation):

        self.model_ = Sequential([
            Conv1D(
                filters=filters, 
                kernel_size=kernel_size,
                strides=1,
                padding="same"
            ),
            AvgPool1D(pool_size=2),
            BatchNormalization(),
            Activation(activation)
        ])
    
    def call(self, input):
        return self.model_(input)

class Linear(Module):

    def __init__(self, units, activation, dp_rate=0.45):

        self.model_ = Sequential([
            Dense(units=units),
            Dropout(rate=dp_rate),
            Activation(activation)
        ])
    
    def call(self, input):
        return self.model_(input)


class Cnn1DRecognizer(Model):

    def __init__(self, input_sh, **kwargs):

        super().__init__(**kwargs)
        self.input_sh = input_sh
        self.model = self.build_model_()
    
    def build_model_(self):

        input = Input(shape=self.input_sh)

        conv = Sequential([
            Conv(filters=128, kernel_size=3, activation="tanh"),
            Conv(filters=64, kernel_size=3, activation="tanh"),
            Conv(filters=32, kernel_size=3, activation="tanh"),
        ])(input)

        linear = Sequential([
            Flatten(),
            Linear(units=128, activation="silu"),
            Linear(units=64, activation="silu"),
            Linear(units=32, activation="silu"),
            Linear(units=3, activation="softmax")
        ])(conv)

        return Model(inputs=input, outputs=linear)

    @function
    def call(self, input):
        return self.model(input)
    
    def compile(self, optimizer, loss_fn):

        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_tracker = Mean(name="model_loss")

    @property
    def metrics(self):
        return [
            self.loss_tracker
        ]

    def train_step(self, data):

        mels, labels = data
        with GradientTape() as gr_tape:

            pred_labels = self.model(mels)
            loss = self.loss_fn(labels, pred_labels)
        
        tr_vars = self.model.trainable_variables
        grads = gr_tape.gradient(loss, tr_vars)
        self.optimizer.apply_gradients(zip(grads, tr_vars))

        self.loss_tracker.update_state(loss)
        return {
            "prediction loss": self.loss_tracker.result()
        }






    
