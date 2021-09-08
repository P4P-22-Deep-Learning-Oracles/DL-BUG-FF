import tensorflow as tf

image = tf.image.decode_jpeg("~/Desktop/test.jpg", channels=1)

imageSize = [20,20]
tf.image.resize(image, imageSize, 'random', 'random')
imageSize = [220]
tf.image.resize(image, [10,10], 'random', 'random')
print(image)