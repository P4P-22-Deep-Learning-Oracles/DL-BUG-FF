import tensorflow as tf
import input_data
import os

#消除Tensorflow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#创建一个神经网络主要有一下4个步骤：
#1.定义神经网络的结构，即计算图
#2.定义损失函数
#3.在会话中，将数据输入进构建的神经网络中，反复优化损失函数，直至得到最优解
#4.将测试集丢入训练好的神经网络进行验证

#learning_rate：用于控制梯度下降的幅度
#batch_size：一次性输入数据个数
#epoch_size：遍历总数
#display_step：打印中间结果的步数
#一个训练batch中的训练数据个数
learning_rate = 0.01
batch_size = 16
epoch_step = 30000
display_step = 1000
BATCH_SIZE = 100  

#定义输入层、隐藏层、输出层神经元
#x：输入层
#y：输出层
x = tf.placeholder("float", [None, 784], name="input")
y = tf.placeholder("float", [None, 10], name="output")

#layer1、layer2代表隐藏层神经元个数
layer1 = 16
layer2 = 32

# 定义神经网络参数w、b
# w中，h1代表的是输入层到第一层隐藏层的权重，因为输入层有784个神经元，隐藏层神经元个数为layer1，
# 所以维度为[784,layer1]。h2同理。out表示第二层隐藏层到输出层的权重。
# b同理。
# tf.Variable是用来定义变量的函数，tf.random_normal是随机生成正太分布函数
w = {
	"h1": tf.Variable(tf.random_normal([784, layer1])),
	"h2": tf.Variable(tf.random_normal([layer1, layer2])),
	"out": tf.Variable(tf.random_normal([layer2, 10]))
}

b = {
	"h1": tf.Variable(tf.random_normal([layer1])),
	"h2": tf.Variable(tf.random_normal([layer2])),
	"out": tf.Variable(tf.random_normal([10]))
}

# 定义神经网络函数
def network(x_input, weights, biases):
	# net = [input]*[w] + b
	net1 = tf.nn.relu(tf.matmul(x_input, weights["h1"]) + biases["h1"])
	net2 = tf.nn.relu(tf.matmul(net1, weights["h2"]) + biases["h2"])
	output = tf.matmul(net2, weights["out"] + biases["out"])
	return output

# 定义损失函数
# pred为预测的数据，即神经网络的输出
pred = network(x, w, b)
# cost即损失函数，tf.reduce_mean是求平均损失，因为一次性输入的是多个（batch_size个）数据。
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
# tf.train.AdamOptimizer是选择的优化器，其作用是最小化cost
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
#init为初始化变量
init = tf.global_variables_initializer()

# 用于检测训练结果
# tf.argmax函数返回的是张量在某一维最大值的索引值，由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签。
# 如果pred的最大值所在的索引值等于类别标签的索引值，表示这个结果分类正确
# tf.equal是TensorFlow中判断两个张量是否相等，返回的是一个布尔型张量，如[True,False,False]。
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

mnist = input_data.read_data_sets('data/', one_hot=True)
# 创建会话，执行神经网络
with tf.Session() as sess:
	# 首先进行初始化sess.run(init)
	sess.run(init)
	test_x, test_y = mnist.test.images, mnist.test.labels

	# 接着遍历epoch_step次所有数据
	for epoch in range(epoch_step):
		xs, ys = mnist.train.next_batch(BATCH_SIZE)
		t, output = sess.run([cost, optimizer], feed_dict={x: xs, y: ys})
		# optimizer.run(feed_dict={x: xs, y: ys})

	print("finish!")

	# 将测试集丢入训练好的神经网络进行验证
	#打印准确率，为测试训练集的结果
	print("accuracy: ", sess.run(tf.argmax(pred, 1), feed_dict = {x: [test_x[0]]}))
