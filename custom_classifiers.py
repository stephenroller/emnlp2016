#!/usr/bin/env python

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_recall_curve
import sklearn.svm
import sklearn.tree
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.linear_model
from sklearn.preprocessing import normalize

class SuperTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features=4):
        self.models = []
        self.linear = sklearn.linear_model.LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced')
        self.n = n_features
        #self.final = sklearn.tree.DecisionTreeClassifier(max_depth=9)
        #self.final = sklearn.naive_bayes.MultinomialNB()
        #self.final = sklearn.naive_bayes.GaussianNB()
        #self.final = sklearn.naive_bayes.BernoulliNB()
        self.final = sklearn.svm.SVC(kernel='rbf', class_weight='balanced', max_iter=2000)
        #kernel='linear', class_weight='balanced', max_iter=2000, verbose=True)
        #self.final = sklearn.svm.LinearSVC(dual=False)

    def _subproj(self, plane, X):
        plane = plane / np.sqrt(plane.dot(plane))
        D = X.shape[1] / 2
        cos = np.sum(np.multiply(X[:,:D], X[:,D:]), axis=1)
        proj1 = X[:,:D].dot(plane)
        proj2 = X[:,D:].dot(plane)
        X1 = normalize(X[:,:D] - np.outer(proj1, plane))
        X2 = normalize(X[:,D:] - np.outer(proj2, plane))
        Xn = np.concatenate([X1, X2], axis=1)

        return np.array([cos, proj1, proj2]), Xn # this one's pretty good
        #return np.array([cos, proj1, proj2, proj2 - proj1]), Xn # this one's pretty good

    def fit(self, X, y):
        self.models = []
        from sklearn.base import clone
        from sklearn.metrics import f1_score
        self.planes = []
        extraction = []
        for i in xrange(self.n):
            D = X.shape[1] / 2
            #self.linear.fit(X[:,:D], y)
            #f1l = f1_score(y, self.linear.predict(X[:,:D]))
            #hyperplane = self.linear.coef_[0]

            #self.linear.fit(X[:,D:], y)
            #f1r = f1_score(y, self.linear.predict(X[:,D:]))

            #if f1l < f1r: hyperplane = self.linear.coef_[0]
            #print max(f1l, f1r)

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
            hyperplane = hyperplane / np.sqrt(hyperplane.dot(hyperplane))
            extraction.append(feats)
            self.planes.append(hyperplane)

        Xe = (np.concatenate(extraction).T)
        #Xet = Xe
        #Xe = np.concatenate([Xe, X], axis=1)
        self.final.fit(Xe, y)
        return self

    def predict(self, X):
        extraction = []
        for p in self.planes:
            feats, X = self._subproj(p, X)
            extraction.append(feats)
        Xe = (np.concatenate(extraction).T)
        #Xet = Xe
        #Xe = np.concatenate([Xe, X], axis=1)
        #return self.nb.predict(Xe)
        return self.final.predict(Xe)

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

def levy_kernel(U, V):
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



