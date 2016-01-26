#!/usr/bin/env python

import numpy as np

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


