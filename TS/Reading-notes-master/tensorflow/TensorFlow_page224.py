# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np 

import matplotlib as mpl 
mpl.use('Agg')
from matplotlib import pyplot as plt 

learn = tf.contrib.learn

HIDDEN_SIZE = 30
NUM_LAYERS = 2
TIMESTEPS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01

def generate_data(seq):
    x = []
    y = []
    print(len(seq) - TIMESTEPS -1)
    for i in range(len(seq) - TIMESTEPS -1):
        x.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

def lstm_model(X,y):
    # 创建深度LSTM，深度为 HIDDEN_SIZE
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
    # 将 lstm_cell 变为多层RNN，层数为NUM_LAYERS
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)
    # 训练rnn，output为输出的结果，_ 返回的是最终的状态
    output,_ = tf.nn.dynamic_rnn(cell, X,dtype=tf.float32)
    # 将output 重塑成 n×HIDDEN_SIZE 的矩阵，即每行属于同一层
    output = tf.reshape(output,[-1, HIDDEN_SIZE])
    # 创建一个全连接层，1 表示输出的维度为1，即做的是 n×HIDDEN_SIZE 的矩阵 和 HIDDEN_SIZE×1的矩阵相乘。None指的是不使用激活函数。
    predictions = tf.contrib.layers.fully_connected(output, 1, None)
    # 重塑 y 和 predictions 
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])
    # 得到均方损失
    loss = tf.losses.mean_squared_error(predictions, labels)
    # 得到训练操作
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)
    return predictions,loss,train_op


test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP

print (test_start)
print (test_end)
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype = np.float32)))
test_X , test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype = np.float32)))

regressor = learn.SKCompat(learn.Estimator(model_fn=lstm_model))
regressor.fit(train_X, train_y, batch_size = BATCH_SIZE, steps = TRAINING_STEPS)
predicted = [[pred] for pred in regressor.predict(test_X)]

rmse = np.sqrt(((predicted - test_y)** 2).mean(axis = 0))
print("Mean Square Error is %f" % rmse[0])


fig = plt.figure()
plot_predicted = plt.plot(predicted, label='predicted')
plot_test = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
fig.savefig('sin.png')