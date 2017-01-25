#!/usr/bin/env python

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_recall_curve
import sklearn.svm
import sklearn.linear_model
from sklearn.preprocessing import normalize

class SuperTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features=4, proj=True, cos=True, incl=True):
        self.models = []
        self.linear = sklearn.linear_model.LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced')
        self.n_features = n_features
        self.proj = proj
        self.cos = cos
        self.incl = incl
        #self.final = sklearn.svm.SVC(kernel='rbf', class_weight='balanced', max_iter=2000)
        self.final = sklearn.linear_model.LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced')

    def _rejection(self, plane, X):
        plane = plane / np.sqrt(plane.dot(plane))
        D = X.shape[1]
        cos = np.sum(np.multiply(X, X), axis=1)
        proj = X.dot(plane)
        rejection = normalize(X - np.outer(proj, plane))
        return proj, rejection

    def _subproj(self, plane, X):
        D = X.shape[1] / 2
        cos = np.sum(np.multiply(X[:,:D], X[:,D:]), axis=1)
        proj1, X1 = self._rejection(plane, X[:,:D])
        proj2, X2 = self._rejection(plane, X[:,D:])
        Xn = np.concatenate([X1, X2], axis=1)

        #return np.array([cos, proj1, proj2]), Xn # this one's pretty good
        features = []
        if self.proj:
            features += [proj1, proj2]
        if self.cos:
            features += [cos]
        if self.incl:
            features += [proj2 - proj1]
        return np.array(features), Xn

    def fit(self, X, y):
        self.models = []
        from sklearn.base import clone
        from sklearn.metrics import f1_score
        self.planes = []
        extraction = []
        for i in xrange(self.n_features):
            D = X.shape[1] / 2
            # copy it for feature extraction purposes
            self.linear.fit(X, y)
            self.models.append(clone(self.linear))
            self.models[-1].coef_ = self.linear.coef_

            lhs = self.linear.coef_[0,:D]
            rhs = self.linear.coef_[0,D:]
            if lhs.dot(lhs) > rhs.dot(rhs):
                hyperplane = lhs
            else:
                hyperplane = rhs
            feats, X = self._subproj(hyperplane, X)
            self.planes.append(hyperplane)
            hyperplane = hyperplane / np.sqrt(hyperplane.dot(hyperplane))
            extraction.append(feats)

        self.coef_ = np.array(self.planes)
        Xe = np.concatenate(extraction).T
        self.final.fit(Xe, y)
        return self

    def predict(self, X):
        extraction = []
        for p in self.planes:
            feats, X = self._subproj(p, X)
            extraction.append(feats)
        Xe = (np.concatenate(extraction).T)
        return self.final.predict(Xe)

    def set_params(self, **kwargs):
        for hp in ['n_features', 'proj', 'cos', 'incl']:
            if hp in kwargs:
                setattr(self, hp, kwargs[hp])
                del kwargs[hp]
        self.final.set_params(**kwargs)

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

def ksim_kernel(U, V):
    # kernel((ul, ur), (vl,  vr)) =
    #   (ul ur * vl vr) ^ (alpha / 2) *
    #   (ul vl * ur vr) ^ (1 - alpha / 2)
    alpha = 0.5

    D = U.shape[1] / 2
    Ul, Ur = U[:,:D], U[:,D:]
    Vl, Vr = V[:,:D], V[:,D:]
    ulur = np.array([np.multiply(Ul, Ur).sum(axis=1)]).clip(0, 1e99)
    vlvr = np.array([np.multiply(Vl, Vr).sum(axis=1)]).clip(0, 1e99)
    ulvl = Ul.dot(Vl.T)
    urvr = Ur.dot(Vr.T)

    a = np.dot(ulur.T, vlvr)
    b = np.multiply(ulvl, urvr).clip(0, 1e99)

    a2 = np.power(a, alpha / 2.)
    b2 = np.power(b, 1 - alpha / 2.)

    retval = np.multiply(a2, b2)
    return retval

