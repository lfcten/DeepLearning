#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/18 13:39
# @Author  : Danxiyang
# @Site    :
# @File    : __init__.py.py
# @Software: PyCharm
"""
dynamic pooling: k_max_pooling
"""
import tensorflow as tf
import numpy as np
a = tf.constant([1, 2, 3, 5, 2, 10, 3, 7], dtype=tf.float32)


def argsort(input_array):
    index = np.argsort(input_array)
    input_array = input_array[index]
    return np.array(input_array).astype(np.float32)


def top_K_index(input_array, k):
    index = np.argsort(input_array)[-k:]
    input_array = input_array[np.sort(index)]
    return np.array(input_array).astype(np.float32)


# def top_K(input_array, k, index_sort=False, sorted=True):
#     if not index_sort:
#         return tf.nn.top_k(input_array, k, sorted=sorted).values
#     else:
#         return tf.py_func(top_K_index, [input_array, k], tf.float32)

k = 3
top_K_index = tf.py_func(top_K_index, [a, k], tf.float32)

# parse_json_op = tf.py_func(argsort, [a], tf.float32)
# b = tf.argmax(input=parse_json_op, dimension=0)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(top_K_index))