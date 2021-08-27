# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 02:17:55 2021

@author: Michael

Buggy code sample 3 as on excel
"""
import numpy as np
import random
import tensorflow as tf
folder = 'D:\\Lab_Project_Files\\TF\\Practice Files\\'
Datainfo = 'dataset_300.txt'
ClassInfo = 'classTrain.txt'

INPUT_WIDTH  = 16
IMAGE_HEIGHT = 12
IMAGE_DEPTH  = 1
IMAGE_PIXELS = INPUT_WIDTH * IMAGE_HEIGHT # 192 = 12*16
NUM_CLASSES  = 8
STEPS         = 500
STEP_VALIDATE = 100
BATCH_SIZE    = 5

def load_data(file1,file2,folder):
    filename1 = folder + file1
    filename2 = folder + file2
    # loading the data file
    x_data = np.loadtxt(filename1, unpack=True)
    x_data = np.transpose(x_data)
    # loading the class information of the data loaded
    y_data = np.loadtxt(filename2, unpack=True)
    y_data = np.transpose(y_data)
    # divide the data in to test and train data
    x_data_train = x_data[np.r_[0:20, 45:65, 90:110, 135:155, 180:200, 225:245, 270:290, 315:335],:]
    x_data_test  = x_data[np.r_[20:45, 65:90, 110:135, 155:180, 200:225, 245:270, 290:315, 335:360], :]
    y_data_train = y_data[np.r_[0:20, 45:65, 90:110, 135:155, 180:200, 225:245, 270:290,  315:335]]
    y_data_test  = y_data[np.r_[20:45, 65:90, 110:135, 155:180, 200:225, 245:270, 290:315, 335:360],:]
    return x_data_train,x_data_test,y_data_train,y_data_test

def reshapedata(data_train,data_test):
    data_train  = np.reshape(data_train, (len(data_train),INPUT_WIDTH,IMAGE_HEIGHT))
    data_test   = np.reshape(data_test,  (len(data_test), INPUT_WIDTH, IMAGE_HEIGHT))
    return data_train,data_test

def batchdata(data,label, batchsize):
    # generate random number required to batch data
    order_num = random.sample(range(1, len(data)), batchsize)
    data_batch = []
    label_batch = []
    for i in range(len(order_num)):
        data_batch.append(data[order_num[i-1]])
        label_batch.append(label[order_num[i-1]])
    return data_batch, label_batch

# CNN trail
def conv_net(x):
    weights = tf.Variable(tf.random_normal([INPUT_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH, NUM_CLASSES]))
    biases = tf.Variable(tf.random_normal([NUM_CLASSES]))
    out = tf.add(tf.matmul(x, weights), biases)
    return out

sess = tf.Session()
# get filelist and labels for training and testing
data_train,data_test,label_train,label_test =         load_data(Datainfo,ClassInfo,folder)
data_train, data_test, = reshapedata(data_train, data_test)

############################ get files for training ####################################################
image_batch, label_batch = batchdata(data_train,label_train,BATCH_SIZE)
# input output placeholders
x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
y_ = tf.placeholder(tf.float32,[None, NUM_CLASSES])
# create the network
y = conv_net( x )
# loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
# train step
train_step   = tf.train.AdamOptimizer( 1e-3 ).minimize( cost )

############################## get files for validataion ###################################################
image_batch_test, label_batch_test = batchdata(data_test,label_test,BATCH_SIZE)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

################ CNN Program ##############################

for i in range(STEPS):
        # checking the accuracy in between.
        if i % STEP_VALIDATE == 0:
            imgs, lbls = sess.run([image_batch_test, label_batch_test])
            print(sess.run(accuracy, feed_dict={x: imgs, y_: lbls}))

        imgs, lbls = sess.run([image_batch, label_batch])
        sess.run(train_step, feed_dict={x: imgs, y_: lbls})

imgs, lbls = sess.run([image_batch_test, label_batch_test])
print(sess.run(accuracy, feed_dict={ x: imgs, y_: lbls}))