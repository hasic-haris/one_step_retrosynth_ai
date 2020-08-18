"""
Author:      Hasic Haris (Phd Student @ Ishida Lab, Department of Computer Science, Tokyo Institute of Technology)
Created on:  February 28th, 2020
Description: This file contains TensorFlow 1.12 neural network initialization functions.
"""
import tensorflow as tf


# Done: 100%
def init_weights(w_shape, layer_index, weight_initializer):
    """ Initialization of the weight values. """

    return tf.Variable(weight_initializer(w_shape), name="weight{}".format(layer_index))


# Done: 100%
def init_bias(b_shape, layer_index, bias_initializer):
    """ Initialization of the bias values. """

    return tf.Variable(tf.constant(bias_initializer, shape=b_shape, name="bias{}".format(layer_index)))
