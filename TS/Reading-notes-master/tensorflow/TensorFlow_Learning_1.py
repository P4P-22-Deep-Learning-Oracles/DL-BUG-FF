import tensorflow as tf

# 定义一个tensorflow的常数
w = tf.constant([[1,2,3],[4,5,6]])

#w:  Tensor("Const:0", shape=(2, 3), dtype=int32)
print("w: ", w)

# a,b shape=[3,]
a = tf.constant([1,2,3], name="a")
b = tf.constant([4,5,6], name="b")
y = a + b
# y:  Tensor("add:0", shape=(3,), dtype=int32)
# 表明y只是一个计算
print("y: ", y)

# 初始化所有变量
init = tf.global_variables_initializer()

# 构建会话
with tf.Session() as sess:
	#执行初始化操作
	sess.run(init)
	#运行节点y
	output = sess.run(y)
	# output:  [5 7 9]
	# 通过会话得出最终的向量结果
	print("output: ", output)