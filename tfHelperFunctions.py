# -*- coding: utf-8 -*-
"""
Created on Sun May 21 17:03:59 2017

@author: Amar Civgin
"""

'''
summary definition and network architecture
'''

import tensorflow as tf

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act = tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal( [input_dim, output_dim] ), name=layer_name+'weights')
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.random_normal( [output_dim] ), name=layer_name+'biases')
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name = layer_name+'activations')
        tf.summary.histogram('activations', activations)
        return activations

def conv_layer(input_tensor, filter_height, filter_width, in_channels, out_channels, strides, padding, layer_name, act = tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('window'):
            conv_filter = tf.Variable(tf.random_normal([filter_height, filter_width, in_channels, out_channels]), name=layer_name+'filter')
            variable_summaries(conv_filter)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.random_normal( [out_channels] ), name=layer_name+'biases')
            variable_summaries(biases)
        with tf.name_scope('W_conv_x_plus_b'):
            convolution = tf.nn.conv2d(input_tensor, conv_filter, strides = strides, padding = padding) + biases
            tf.summary.histogram('convolution', convolution)
        activatedConv = act(convolution, name = layer_name + 'activated_convolution')
        tf.summary.histogram('activated_convolution', activatedConv)
        return activatedConv

def max_pool_layer(input_tensor, ksize, strides, padding, layer_name):
    with tf.name_scope(layer_name):
        max_pool = tf.nn.max_pool(input_tensor, ksize = ksize, strides = strides, padding = padding, name = layer_name + 'max_pooling')
        tf.summary.histogram('max_pool', max_pool)
        return max_pool

print('Successfully imported tfHelperFunctions')