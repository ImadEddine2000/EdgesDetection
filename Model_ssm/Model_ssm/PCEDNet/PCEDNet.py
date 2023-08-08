import tensorflow as tf


def generate_input(x):
    return [x[:, i, :] for i in range(x.shape[1])]


class PCEDNet:

    def __init__(self):
        pass

    def create_tree(self, i, l):
        if i == 0:
            input1 = tf.keras.layers.Input(shape=(6))
            input2 = tf.keras.layers.Input(shape=(6))
            l.extend([input1, input2])
        else:
            input1 = self.create_tree(i - 1, l)
            input2 = self.create_tree(i - 1, l)
        return tf.keras.layers.Dense(6, activation="selu", use_bias=True)(tf.keras.layers.Concatenate(axis=-1)
                                                                          ([input1, input2]))

    def build(self):
        l = []
        input1 = self.create_tree(2, l)
        input2 = self.create_tree(2, l)
        dense1 = tf.keras.layers.Dense(16, activation="selu", use_bias=True)(
            tf.keras.layers.Concatenate(axis=-1)([input1, input2]))
        dense1 = tf.keras.layers.BatchNormalization()(dense1)
        dense2 = tf.keras.layers.Dense(16, activation="selu", use_bias=True)(dense1)
        dense2 = tf.keras.layers.BatchNormalization()(dense2)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(dense2)
        self.model = tf.keras.models.Model(inputs=l, outputs=output)
        return self.model