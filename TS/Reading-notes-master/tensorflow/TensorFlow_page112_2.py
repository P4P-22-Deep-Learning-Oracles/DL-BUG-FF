import tensorflow as tf 

saver = tf.train.import_meta_graph("./model/page111/model.ckpt.meta")

with tf.Session() as sess:
	saver.restore(sess, "./model/page111/model.ckpt")
	print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))