import keras
from keras import layers

NUM_OBSEREVATIONS = 16
NUM_ACTIONS = 4


def create_q_model():

    x = keras.Input(shape=[NUM_OBSEREVATIONS], dtype="float32")
    y = layers.Dense(NUM_OBSEREVATIONS)(x)
    y = layers.Dense(64, activation="relu")(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Dense(64, activation="relu")(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Dense(64, activation="relu")(y)
    y = layers.Dense(NUM_ACTIONS)(y)

    return keras.Model(x, y)
