#!/usr/bin/env python

import numpy as np

import keras
import keras.backend as K
from keras import initializations
from keras import activations
from keras.models import Sequential
from keras.layers.core import Layer
from keras.regularizers import l2
from keras.optimizers import SGD

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_recall_curve

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.threshold_ = 0.0

    def fit(self, X, y):
        feature = X[:,0]
        p, r, t = precision_recall_curve(y, feature)
        #nonzero = (p > 0) & (r > 0)
        #p, r, t = p[nonzero], r[nonzero], t[nonzero[1:]]
        f1 = np.divide(2 * np.multiply(p, r), p + r)
        f1[np.isnan(f1)] = -1.0
        self.threshold_ = t[f1.argmax()]

    def predict(self, X):
        feature = X[:,0]
        return feature >= self.threshold_

    def predict_proba(self, X):
        return np.concatenate([1-X, X], axis=1)

class BdsmLayer(Layer):
    def __init__(self, input_dim, hidden_dim, init='glorot_uniform', activation='sigmoid'):
        self.D = input_dim
        self.input_dim = self.D * 2
        self.H = hidden_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = 1

        self.W = self.init((self.D, self.H))
        self.b = K.zeros((self.H,))
        self.W_reg = l2(1e-5)
        self.W_reg.set_param(self.W)
        self.regularizers = [self.W_reg]

        self.params = [self.W, self.b]

        super(BdsmLayer, self).__init__(input_shape=(self.input_dim,))

    def get_output(self, train=False):
        X = self.get_input(train)

        lhs = X[:,:self.D]
        rhs = X[:,self.D:]

        lhs2 = self.activation(K.dot(lhs, self.W) + self.b)
        rhs2 = self.activation(K.dot(rhs, self.W) + self.b)

        if True or train:
            upper = K.maximum(1-rhs2, lhs2)
            lower = K.min(upper, axis=-1)
            return K.permute_dimensions(lower, (0, 'x'))
        else:
            lhsrnd = K.round(lhs2)
            rhsrnd = K.round(rhs2)
            num = K.sum(lhsrnd * rhsrnd, axis=-1)
            from keras.backend.common import _EPSILON
            den = K.clip(K.sum(lhsrnd, axis=-1), _EPSILON, self.H)
            frac = num / den
            return K.permute_dimensions(frac, (0, 'x'))




class BDSM(BaseEstimator, ClassifierMixin):
    def __init__(self, H=132):
        D = 300
        self.H = H
        self.model = Sequential()
        self.model.add(BdsmLayer(D, self.H))
        sgd = SGD(lr=0.01, momentum=0.99)
        self.model.compile(loss='mse', optimizer=sgd)
        self.inits = [p.get_value() for p in self.model.params]
        self.eta = 0.5

    def fit(self, X, y):
        y = y.astype(np.int)
        # need to reset the model
        for p, w in zip(self.model.params, self.inits):
            p.set_value(w)
            pass
        self.model.fit(X, y, verbose=1, batch_size=32, nb_epoch=100, shuffle=True)
        # okay we fit the model, but we still need the threshold
        #values = [p.get_value() for p in self.model.params]
        values = self.model.predict(X)[:,0]
        p, r, t = precision_recall_curve(y, values)
        f1 = np.divide(2 * np.multiply(p, r), p + r)
        f1[np.isnan(f1)] = -1.0
        self.eta = t[f1.argmax()]
        print "eta = %.3f, f1[train] = %.3f" % (self.eta, f1.max())
        return self

    def predict(self, X):
        probas = self.predict_proba(X)
        return probas[:,1] > self.eta

    def predict_proba(self, X):
        probas = self.model.predict(X)
        return np.concatenate([1-probas, probas], axis=1)

