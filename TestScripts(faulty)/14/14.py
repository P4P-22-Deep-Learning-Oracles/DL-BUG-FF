# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 02:17:55 2021

@author: Michael

Buggy code sample 14 as on excel
"""
import tensorflow as tf
K = 10

lchild = tf.placeholder(tf.float32, shape=(K))
rchild = tf.placeholder(tf.float32, shape=(K))
parent = tf.nn.tanh(tf.add(lchild, rchild))

input = [ tf.Variable(tf.random_normal([K])),
          tf.Variable(tf.random_normal([K])) ]

with tf.Session() as sess :
    print(sess.run([parent], feed_dict={ lchild: input[0], rchild: input[1] }))