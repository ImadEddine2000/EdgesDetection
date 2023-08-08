import tensorflow as tf
import numpy as np


def generate_input(x):
    return x


class NN(tf.keras.layers.Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super(NN, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(d_ff, activation="selu",
                                            kernel_initializer=tf.keras.initializers.lecun_uniform())
        self.dense2 = tf.keras.layers.Dense(d_model, activation="selu",
                                            kernel_initializer=tf.keras.initializers.lecun_uniform())

    def call(self, x):
        return self.dense2(self.dense1(x))


class AddNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, sublayer_x):
        return self.layer_norm((x + sublayer_x))


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(h, d_k, d_v, dropout=rate)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = NN(d_ff, d_model)

    def call(self, x, padding_mask, training):
        multihead_output = self.multihead_attention(x, x, padding_mask, return_attention_scores=False)
        multihead_output = self.dropout1(multihead_output, training=training)
        addnorm_output = self.add_norm1(x, multihead_output)
        return self.feed_forward(addnorm_output)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_scales, sequence_length, h, d_k, d_v, d_ff, d_model, n, rate, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_ff, d_model, rate) for _ in range(n)]

    def call(self, input_scales, padding_mask, training):
        x = self.dropout(input_scales, training=training)
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)
        return x

class TssmNet:

    def __init__(self, input_shape = (16, 6), h=4, d_k=32, d_v=32, d_ff=12, d_model=16, n=1, dropout_rate=0.1):
        self.activation_func = "selu"
        self.activation_func2 = "sigmoid"
        self.input_1 = tf.keras.layers.Input(shape=input_shape)
        self.encoder = Encoder(input_shape[0], input_shape[0], h, d_k, d_v, d_ff, d_model ,n, dropout_rate)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid", use_bias=True, bias_initializer=tf.keras.initializers.lecun_uniform())

    def build(self):
        input_ = self.input_1
        output = self.encoder(input_, None, True)
        output = self.flatten(output)
        output = self.dense(output)
        self.model = tf.keras.models.Model(inputs=input_, outputs=output)
        return self.model
