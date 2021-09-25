# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 
import tensorflow as tf 

iamge_raw_data = tf.gfile.FastGFile("./data/1.jpg", "rb").read()
with tf.Session() as sess:
	img_data = tf.image.decode_jpeg(iamge_raw_data)
	print(img_data.eval())

	plt.imshow(img_data.eval())
	plt.show()

	img_data = tf.image.convert_image_dtype(img_data, dtype = tf.float32)

	encoded_image = tf.image.encode_jpeg(tf.cast(img_data, tf.uint8))
	with tf.gfile.GFile("./data/output", 'wb') as f:
		f.write(encoded_image.eval())
