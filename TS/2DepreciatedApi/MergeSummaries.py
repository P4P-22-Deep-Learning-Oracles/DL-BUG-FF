import tensorflow as tf

accuracy_summary = tf.scalar_summary("accuracy", 0.345)
loss_summary = tf.scalar_summary("loss", 0.345)

merged = tf.merge_summary([accuracy_summary, loss_summary])
merged = tf.merge_all_summaries()