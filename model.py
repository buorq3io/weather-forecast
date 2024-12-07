import numpy as np
from keras import models, layers, activations, regularizers


def get_model(n_inputs):
    signal = layers.Input(shape=(72, n_inputs), dtype=np.float32)
    x = layers.LSTM(64, activation=activations.tanh, kernel_regularizer=regularizers.l2(1e-2))(signal)
    x = layers.Dense(64, activation=activations.relu, kernel_regularizer=regularizers.l2(1e-2))(x)
    x = layers.Dense(32, activation=activations.relu, kernel_regularizer=regularizers.l2(1e-2))(x)
    output = layers.Dense(1, activation=activations.linear, dtype=np.float32)(x)

    model = models.Model(signal, output)
    return model


if __name__ == '__main__':
    print(get_model(8).summary())
