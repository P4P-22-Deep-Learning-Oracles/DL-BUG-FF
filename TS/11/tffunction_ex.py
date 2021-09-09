import tensorflow as tf

writer = tf.summary.create_file_writer("/tmp/mylogs/tf_function")

@tf.function
def my_func(step):
    with writer.as_default():
        # other model code would go here
        tf.summary.scalar("my_metric", 0.5, step=step)


for step in range(100):
    my_func(step)
    writer.flush()
