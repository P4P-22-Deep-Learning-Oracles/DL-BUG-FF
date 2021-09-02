# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 02:17:55 2021

@author: Michael

Buggy code sample 9 as on excel
"""
import tensorflow as tf
import numpy as np

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

class NeuralNet:
    def __init__(self, hidden):
        self.hidden = hidden

    def __del__(self):
        self.sess.close()

    def fit(self, X, y):
        _X = tf.placeholder('float', [None, None])
        _y = tf.placeholder('float', [None, 1])

        w0 = init_weights([X.shape[1], self.hidden])
        b0 = tf.Variable(tf.zeros([self.hidden]))
        w1 = init_weights([self.hidden, 1])
        b1 = tf.Variable(tf.zeros([1]))

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        h = tf.nn.sigmoid(tf.matmul(_X, w0) + b0)
        self.yp = tf.nn.sigmoid(tf.matmul(h, w1) + b1)

        C = tf.reduce_mean(tf.square(self.yp - y))
        o = tf.train.GradientDescentOptimizer(0.5).minimize(C)

        correct = tf.equal(tf.argmax(_y, 1), tf.argmax(self.yp, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        tf.scalar_summary("accuracy", accuracy)
        tf.scalar_summary("loss", C)

        merged = tf.merge_all_summaries()
        import shutil
        shutil.rmtree('logs')
        writer = tf.train.SummaryWriter('logs', self.sess.graph_def)

        for i in xrange(1000+1):
            if i % 100 == 0:
                res = self.sess.run([o, merged], feed_dict={_X: X, _y: y})
            else:
                self.sess.run(o, feed_dict={_X: X, _y: y})
        return self

    def predict(self, X):
        yp = self.sess.run(self.yp, feed_dict={_X: X})
        return (yp >= 0.5).astype(int)


X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0],[1],[1],[0]])

m = NeuralNet(10)
m.fit(X, y)
yp = m.predict(X)[:, 0]
print(accuracy_score(y, yp))


