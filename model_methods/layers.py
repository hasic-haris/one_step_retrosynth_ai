"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  February 28th, 2020
Description: This file contains the TensorFlow 1.12. neural network layer definitions.
"""

import tensorflow as tf

from model_methods.initializations import init_weights, init_bias


def fully_connected_layer(layer_index, layer_input, input_size, output_size, activation_fcn=None,
                          weight_init=tf.initializers.glorot_normal(), bias_init=0.01):
    """ Constructs a fully connected layer. """

    w = init_weights([input_size, output_size], layer_index, weight_init)
    b = init_bias([output_size], layer_index, bias_init)

    layer_output = tf.add(tf.matmul(layer_input, w), b)

    tf.summary.histogram("weights", w)
    tf.summary.histogram("bias", b)

    if activation_fcn is None:
        return layer_output
    else:
        return activation_fcn(layer_output)


def highway_layer(layer_index, layer_input, input_size, output_size, activation_fcn=None,
                  weight_init=tf.initializers.glorot_normal(), bias_init=0.01, carry_bias_init=-20.0):
    """ Constructs a highway layer. """

    # Step 1: Define weights and biases for the activation gate.
    w = init_weights([input_size, output_size], layer_index, weight_init)
    b = init_bias([input_size], layer_index, bias_init)

    # Step 2: Define weights and biases for the transform gate.
    w_t = init_weights([input_size, output_size], layer_index, weight_init)
    b_t = init_bias([input_size], layer_index, carry_bias_init)

    # Step 3: Calculate activation, transform and carry gate.
    h = activation_fcn(tf.matmul(layer_input, w) + b, name="input_gate")
    t = tf.nn.sigmoid(tf.matmul(layer_input, w_t) + b_t, name="transform_gate")
    c = tf.subtract(1.0, t, name="carry_gate")

    # Step 4: Compute the output from the highway fully connected layer
    layer_output = tf.add(tf.multiply(h, t), tf.multiply(layer_input, c), name="output_highway")

    tf.summary.histogram("activation_gate_weights", w)
    tf.summary.histogram("activation_gate_bias", b)

    tf.summary.histogram("transform_gate_weights", w_t)
    tf.summary.histogram("transform_gate_bias", b_t)

    if activation_fcn is None:
        return layer_output
    else:
        return activation_fcn(layer_output)
