import tensorflow as tf
import numpy as np

def generate_input(x):
    shape = x.shape
    a = int(np.sqrt(shape[1]))
    return x.reshape((shape[0], a, a, shape[2]))

class BatchShufflingLayer(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        if training:
            batch_size = tf.shape(inputs)[0]
            shuffled_indices = tf.random.shuffle(tf.range(batch_size))
            shuffled_inputs = tf.gather(inputs, shuffled_indices)
            return shuffled_inputs
        return inputs

class CNNGM:

    def __init__(self, input_shape, activation_function="selu"):
        self.activation_func = activation_function
        self.input_1 = tf.keras.layers.Input(shape=input_shape)

        self.batchShuffle = BatchShufflingLayer()

        self.conv2D_1 = tf.keras.layers.Conv2D(64, (5, 5), activation=self.activation_func , use_bias=True,
                                             kernel_initializer=tf.keras.initializers.lecun_uniform(),
                                             bias_initializer=tf.keras.initializers.lecun_uniform())
        self.conv2D_2 = tf.keras.layers.Conv2D(128, (10, 10), activation=self.activation_func , use_bias=True,
                                             kernel_initializer=tf.keras.initializers.lecun_uniform(),
                                             bias_initializer=tf.keras.initializers.lecun_uniform())
        self.conv2D_3 = tf.keras.layers.Conv2D(256, (20, 20), activation=self.activation_func , use_bias=True,
                                             kernel_initializer=tf.keras.initializers.lecun_uniform(),
                                             bias_initializer=tf.keras.initializers.lecun_uniform())

        self.activation_func = activation_function
        self.activation_func2 = "sigmoid"

        self.dense_1 = tf.keras.layers.Dense(256, activation=self.activation_func , use_bias=True,
                                             kernel_initializer=tf.keras.initializers.lecun_uniform(),
                                             bias_initializer=tf.keras.initializers.lecun_uniform())

        self.dense_2 = tf.keras.layers.Dense(128, activation=self.activation_func , use_bias=True,
                                             kernel_initializer=tf.keras.initializers.lecun_uniform(),
                                             bias_initializer=tf.keras.initializers.lecun_uniform())

        self.dense_3 = tf.keras.layers.Dense(64, activation=self.activation_func , use_bias=True,
                                             kernel_initializer=tf.keras.initializers.lecun_uniform(),
                                             bias_initializer=tf.keras.initializers.lecun_uniform())
        self.dense_4 = tf.keras.layers.Dense(1, activation=self.activation_func2 , use_bias=True,
                                             kernel_initializer=tf.keras.initializers.lecun_uniform(),
                                             bias_initializer=tf.keras.initializers.lecun_uniform())

    def build(self):
        input__ = self.input_1
        input_ = self.batchShuffle(input__)
        concat = tf.keras.layers.Concatenate()([tf.keras.layers.Flatten()(tf.keras.layers.MaxPool2D((2, 2))(tf.keras.layers.Dropout(0.1)(conv2D(input_)))) for conv2D in [self.conv2D_1, self.conv2D_2, self.conv2D_3]])
        output = tf.keras.layers.Dropout(0.1)(concat)
        output = self.dense_1(output)
        output = self.dense_2(output)
        output = self.dense_3(output)
        output = self.dense_4(output)
        self.model = tf.keras.models.Model(inputs=input__, outputs=output)
        return self.model

