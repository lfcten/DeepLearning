#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/20 10:34
# @Author  : Danxiyang
# @Site    :
# @File    : test_per_dim_conv.py
# @Software: PyCharm
import tensorflow as tf

embedding_size = 5
batch_size = 1
sentence_length = 10
filter_size = 3
def per_dim_conv_layer(x, w, bias):
    """
    @:param
    :param filter_size: 卷积核大小
    :param x: 输入
    :param w: 卷积核权重
    :param bias: 偏置
    :return:
    """
    input_unstack = tf.unstack(x, axis=2)
    input_unstack = tf.concat([tf.zeros([embedding_size, batch_size, filter_size-1, 1], tf.float32), input_unstack], axis= 2)
    w_unstack = tf.unstack(w, axis=1)
    b_unstack = tf.unstack(bias, axis=1)
    convs = []
    with tf.name_scope("per_dim_conv"):
        for i in range(embedding_size):
            conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="SAME") + b_unstack[i]) #[batch_size, k1+ws2-1, num_filters[1]]
            convs.append(conv)
    conv = tf.stack(convs, axis=2)
        #[batch_size, k1+ws-1, embed_size, num_filters]
    return conv



x = tf.Variable(tf.random_uniform([batch_size, sentence_length, embedding_size, 1], -1.0, 1.0))
w = tf.Variable(tf.truncated_normal([3, embedding_size, 1, 3], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[3, embedding_size]))

out = per_dim_conv_layer(x, w, b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(out.eval())