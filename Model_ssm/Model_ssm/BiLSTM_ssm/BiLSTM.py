import tensorflow as tf
import os
import numpy as np

def generate_input(x):
    return x

class BiLSTM:

    def __init__(self, unit = 16):
        self.activation_func = "selu"
        self.activation_func2 = "sigmoid"
        self.forward_layer = tf.keras.layers.LSTM(unit, return_sequences=False)
        self.backward_layer = tf.keras.layers.LSTM(unit, go_backwards=True, return_sequences=False)
        self.bidirectional = tf.keras.layers.Bidirectional(self.forward_layer, backward_layer=self.backward_layer,
                                                           merge_mode='concat')
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid", use_bias=True, bias_initializer=tf.keras.initializers.lecun_uniform())

    def build(self):
        input_ = tf.keras.layers.Input(shape=(16, 6))
        output = self.bidirectional(input_)
        output = self.dropout(output)
        output = self.dense(output)
        self.model = tf.keras.models.Model(inputs=input_, outputs=output)
        return self.model