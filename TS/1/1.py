# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 02:17:55 2021

@author: Michael

Buggy code sample 1 as on excel sheet
"""
import tensorflow as tf

image = tf.image.decode_jpeg("~/Desktop/test.jpg", channels=1)
print(image)
