import os

import numpy as numpy
import pandas as pandas
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

#############
# MODELDEFS #
#############


class CustomLayer(tf.Module):
    def __init__(self, batch_size, in_size, out_size):
        super(CustomLayer, self).__init__()
        bs = batch_size

        self.weight = tf.Variable(
            self.init_weight(bs, in_size, out_size), dtype=tf.float32
        )
        self.bias = tf.Variable(self.init_bias(bs, out_size), dtype=tf.float32)
        #########################
        # # DO OTHER STUFF HERE #
        #########################

    def init_weight(self, batch_size, size1, size2):
        initializer = tf.keras.initializers.GlorotNormal()
        init = initializer(shape=(size1, size2))
        return [init for b in range(batch_size)]

    def init_bias(self, batch_size, size):
        initializer = tf.keras.initializers.Zeros()
        init = initializer(shape=(size))
        return [init for b in range(batch_size)]

    def __call__(self, inputs):
        y = 1
        ################################
        # DO YOUR CALL CALCULATIONS HERE
        ################################

        return tf.keras.activations.linear(y)


class DenseLayer(tf.Module):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_init=tf.keras.initializers.Orthogonal(),
        bias_init=tf.keras.initializers.Zeros(),
    ):
        super(DenseLayer, self).__init__()
        self.kernel = tf.Variable(
            self.init_weights(in_size, out_size, kernel_init), dtype=tf.float32
        )
        self.bias = tf.Variable(self.init_bias(out_size, bias_init), dtype=tf.float32)

    def init_weights(self, size1, size2, initializer):
        init = initializer(shape=(size1, size2))
        return [init]

    def init_bias(self, size, initializer):
        init = initializer(shape=(size))
        return [init]

    def __call__(self, input):
        return tf.matmul(input, self.kernel) + self.bias


class SubModel(tf.Module):
    def __init__(self, n_layers, hidden_dim, batch_size):
        super(SubModel, self).__init__()
        self.layers = []
        #################################
        # construct model parameters
        ################################

    def forward(self, input):
        x = self.layers[0](input)
        for l in self.layers[1:]:
            x = l(x)
        return x


class MainModel(tf.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.model = SubModel()

    def forward(self, input):
        sequence_length = input.shape[1]
        input = tf.cast(tf.expand_dims(input, axis=-1), dtype=tf.float32)
        output_sequence = tf.TensorArray(
            dtype=tf.float32, size=sequence_length, clear_after_read=False
        )

        for i in range(sequence_length):
            model_in = tf.concat(())  # do stuff
            ################################################################
            # code goes here                                               #
            ################################################################
            output = 1  # get output
            output_sequence = output_sequence.write(
                i, output
            )  # this writes output to sequence

        output_sequence = output_sequence.stack()
        return output_sequence
