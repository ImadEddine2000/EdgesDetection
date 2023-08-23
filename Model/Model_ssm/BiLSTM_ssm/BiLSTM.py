import tensorflow as tf
import os
import numpy as np

def generate_input(x):
    return x
def generate_input14(x):
    return x[:, :14 ,:]
def generate_input12(x):
    return x[:, :12 ,:]
def generate_input10(x):
    return x[:, :10 ,:]

def generate_input8(x):
    return x[:, :8 ,:]

def generate_input6(x):
    return x[:, :6 ,:]

class BiLSTM:

    def __init__(self, unit = 16):
        self.activation_func = "selu"
        self.activation_func2 = "sigmoid"
        self.forward_layer = tf.keras.layers.LSTM(unit, return_sequences=False)
        self.backward_layer = tf.keras.layers.LSTM(unit, go_backwards=True, return_sequences=False)
        self.bidirectional = tf.keras.layers.Bidirectional(self.forward_layer, backward_layer=self.backward_layer, merge_mode='concat')
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense_1 = tf.keras.layers.Dense(12, activation=self.activation_func, kernel_initializer=tf.keras.initializers.lecun_uniform(), use_bias=True, bias_initializer=tf.keras.initializers.lecun_uniform())
        self.dense_2 = tf.keras.layers.Dense(1, activation=self.activation_func2, kernel_initializer=tf.keras.initializers.lecun_uniform(), use_bias=True, bias_initializer=tf.keras.initializers.lecun_uniform())

    def build(self):
        input_ = tf.keras.layers.Input(shape=(16, 6))
        output = self.bidirectional(input_)
        output = self.dropout(output)
        output = self.dense_1(output)
        output = self.dense_2(output)
        self.model = tf.keras.models.Model(inputs=input_, outputs=output)
        return self.model