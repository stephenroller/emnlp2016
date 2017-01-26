#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle


BATCH_SIZE = 32
DROPOUT = 0.5
EPOCHS = 25
LEARNING_RATE = 0.1
MOMENTUM_RATE = 0.9
REG_RATE = 0.0

def glorot(shape):
    v = np.sqrt(6./np.sum(shape))
    return tf.random_uniform(shape, -v, v)

class MirrorClassifier(BaseEstimator, ClassifierMixin):
    sess = None

    def __init__(self, k=300, n_features=50):
        self.k = k
        self.n = n_features

    def _number_outputs(self, labels):
        if labels.max() == 1:
            return 1
        else:
            return labels.max() + 1

    def _setup(self):
        if self.sess:
            self.sess.close()
            self.sess = None

        self.X_ = tf.placeholder(tf.float32, shape=[None, self.k * 2], name="vectors")
        self.Y_ = tf.placeholder(tf.int32, [None], name="labels")
        self.dropout_ = tf.placeholder(tf.float32, [], name="dropout")
        self.weights_ = tf.placeholder(tf.float32, [None, 1], name="weights")

        left0 = self.X_[:,:self.k]
        rite0 = self.X_[:,self.k:]

        self.W1 = tf.Variable(glorot([self.k, self.n]), name="W1")
        self.W1do = tf.nn.dropout(self.W1, keep_prob=(1. - self.dropout_))
        left1 = tf.nn.relu(tf.matmul(left0, self.W1do))
        rite1 = tf.nn.relu(tf.matmul(rite0, self.W1do))
        incl = left1 - rite1
        sims = tf.reduce_sum(tf.mul(left0, rite0), 1, True)
        layer1 = tf.concat(1, [left1, rite1])
        h = layer1.get_shape()[1].value

        self.W2 = tf.Variable(glorot([h, h]), name="W2")
        self.B2 = tf.Variable(tf.zeros(h), name="bias2")
        self.W2do = tf.nn.dropout(self.W2, keep_prob=(1. - self.dropout_))
        layer2 = (tf.nn.bias_add(tf.matmul(layer1, self.W2do), self.B2))

        #self.W3 = tf.Variable(glorot([h, self.o]), name="W2")
        #self.W3do = tf.nn.dropout(self.W3, keep_prob=(1. - self.dropout_))
        #self.B3 = tf.Variable(tf.zeros(self.o), name="bias3")
        #layer3 = (tf.nn.bias_add(tf.matmul(layer2, self.W3do), self.B3))

        self.output = layer2
        if self.o == 1:
            self.probas = tf.nn.sigmoid(self.output)
            self.obj = tf.nn.sigmoid_cross_entropy_with_logits(tf.reshape(self.output, [-1]), tf.to_int32(self.Y_))
        else:
            self.probas = tf.nn.softmax(self.output)
            self.obj = tf.nn.sparse_softmax_cross_entropy_with_logits(self.output, self.Y_)
        self.obj = tf.reduce_mean(tf.mul(self.weights_, self.obj))
        #self.opt = tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM_RATE).minimize(self.obj)
        self.opt = tf.train.AdamOptimizer().minimize(self.obj)

        self.sess = tf.Session()

    @property
    def coef_(self):
        if self.sess:
            return self.sess.run(self.W1).T
        return None

    def fit(self, X, y):
        if not (self.sess and self._number_outputs(y) == self.o):
            self.o = self._number_outputs(y)
            self._setup()

        self.sess.run(tf.initialize_all_variables())

        weights = compute_class_weight('balanced', np.arange(y.max() + 1), y)[y.astype(np.int32)].reshape(-1, 1)
        X = X.astype(np.float32)
        full_data_fd = {
            self.X_: X,
            self.Y_: y,
            self.weights_: weights,
            self.dropout_: 0.0,
        }
        for epoch in xrange(1, EPOCHS+1):
            X, y, weights = shuffle(X, y, weights)
            for mb in xrange(len(X) / BATCH_SIZE):
                i, j = mb * BATCH_SIZE, max(len(X), (mb + 1) * BATCH_SIZE)
                fd = {
                    self.X_: X[i:j],
                    self.Y_: y[i:j],
                    self.weights_: weights[i:j],
                    self.dropout_: DROPOUT,
                }
                _ = self.sess.run([self.obj, self.opt], fd)
            loss = self.sess.run(self.obj, full_data_fd)
            print "%4d loss: %.3f" % (epoch, loss)
        return self

    def predict_proba(self, X):
        fd = {self.X_: X.astype(np.float32), self.dropout_: 0.0}
        return self.sess.run(self.probas, fd)

    def predict(self, X):
        if self.o == 1:
            return self.predict_proba(X) >= 0.5
        else:
            return self.predict_proba(X).argmax(axis=1)

    def set_params(self, **kwargs):
        # todo, uninialize the model
        pass

