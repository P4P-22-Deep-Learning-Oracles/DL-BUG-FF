import tensorflow as tf 

v = tf.Variable(0, dtype = tf.float32, name = 'v')

for variable in tf.all_variables():
	print(variable.name)

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.all_variables())

for variable in tf.all_variables():
	print(variable.name)

saver = tf.train.Saver()
with tf.Session() as sess:
	init_op = tf.initialize_all_variables()
	sess.run(init_op)

	sess.run(tf.assign(v, 10))
	sess.run(maintain_averages_op)

	saver.save(sess, "./model/page113/model.ckpt")
	print(sess.run([v, ema.average(v)]))