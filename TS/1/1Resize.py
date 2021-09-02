import tensorflow as tf

image = tf.image.decode_jpeg("~/Desktop/test.jpg", channels=1)

tf.image.resize(image, [20, 20], 'random', 'random')
print(image)