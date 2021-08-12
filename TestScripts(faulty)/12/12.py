# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 02:17:55 2021

@author: Michael

Buggy code sample 12 as on excel
"""
def MapToSequences(x):
    # x.get_shape().as_list() = [64, 1, None, 512]
    x = tf.squeeze(x)
    # tf.shape(x) = [None, None, None], at runtime would be [64, seqlen, 512]
    x = tf.transpose(x, perm=[1, 0, 2])
    # [seqlen, 64, 512]
    # Here I'd like to unstack with seqlen as num
    x = tf.unstack(x) # Cannot infer num from shape (?, ?, ?)
    return x