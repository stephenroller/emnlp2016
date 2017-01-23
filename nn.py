import keras
import keras.backend as K
import numpy as np
from keras import initializations
from keras import activations
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Dropout, Activation
from keras.regularizers import l2, Regularizer
from keras.optimizers import SGD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_recall_curve, f1_score
import theano.tensor as T


class MirrorLayer(Layer):
    def __init__(self, input_dim, H):
        self.D = input_dim
        self.H = H
        self.input_dim = self.D * 2
        self.init = initializations.get('glorot_uniform')
        self.W = self.init((self.D, self.H))
        #self.b = K.zeros((self.H,))
        self.W_reg = l2(1e-3)
        self.W_reg.set_param(self.W)
        #self.b_reg = l2(1e-3)
        #self.b_reg.set_param(self.b)

        #self.regularizers = [self.W_reg, self.b_reg]
        #self.params = [self.W, self.b]
        self.regularizers = [self.W_reg]
        self.params = [self.W]


        super(MirrorLayer, self).__init__(input_shape=(self.input_dim,))

    def get_output(self, train=False):
        X = self.get_input(train)

        lhs = X[:,:self.D]
        rhs = X[:,self.D:]

        lhso = K.dot(lhs, self.W)
        rhso = K.dot(rhs, self.W)

        incl = lhso - rhso
        inner = K.sum(lhs * rhs, axis=1, keepdims=True)

        return K.concatenate((lhso, rhso, incl, inner), axis=1)
    
    @property
    def output_shape(self):
        return (None, 3 * self.H + 1)

from sklearn.utils.class_weight import compute_class_weight
#from keras.utils.np_utils import to_categorical
to_categorical = lambda x: x

class Stephen(BaseEstimator, ClassifierMixin):
    def __init__(self, H=4):
        D = 300
        sgd = SGD(lr=0.01, momentum=0.9)
        #sgd = 'adadelta'
        self.model = Sequential()
        self.model.add(MirrorLayer(D, H))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(H*3+1, W_regularizer=l2(1e-3)))
        self.model.add(Dropout(0.5))
        self.model.add(Activation('tanh'))
        #self.model.add(Dense(2, activation='softmax', W_regularizer=l2(1e-3)))
        #self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
        self.model.add(Dense(1, activation='sigmoid', W_regularizer=l2(1e-2)))
        self.model.compile(loss='binary_crossentropy', optimizer=sgd)
        self.inits = [p.get_value() for p in self.model.params]
        self.eta = 0.5

    def fit(self, X, y, validation=None):
        #y = y.astype(np.float32)
        # need to reset the model
        klass_weight = compute_class_weight('balanced', np.array([0, 1]), y)
        for p, w in zip(self.model.params, self.inits):
            p.set_value(w)
        if validation:
            validation_X, validation_y = validation
            validation = (validation_X, to_categorical(validation_y))
        self.model.fit(X, to_categorical(y), verbose=1, batch_size=64, nb_epoch=200, shuffle=True, validation_data=validation, class_weight=klass_weight)
        return self

    def predict(self, X):
        probas = self.predict_proba(X)
        return probas[:,1] > self.eta

    def predict_proba(self, X):
        probas = self.model.predict(X)
        return np.concatenate([1-probas, probas], axis=1)


