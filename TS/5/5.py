# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 02:17:55 2021

@author: Michael

Buggy code sample 5 as on excel
"""
image_string = tf.read_file(filename)
image_decoded = tf.image.decode_jpeg(image_string, channels=3)