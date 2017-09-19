#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/19 17:24
# @Author  : Danxiyang
# @Site    : 
# @File    : argsort.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np


a = tf.constant([1, 2, 3, 5, 2, 10, 3, 7], dtype=tf.float32)


def argsort(input_array):
    index = np.argsort(input_array)
    # input_array = input_array[index]
    return np.array(index).astype(np.float32)


argsort = tf.py_func(argsort, [a], tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(argsort, feed_dict={a: a.eval()}))