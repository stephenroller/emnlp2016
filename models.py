#!/usr/bin/env python

import numpy as np
from sklearn import svm, linear_model, dummy

from custom_classifiers import ThresholdClassifier

SETUPS = {
    # baseline most common
    'baseline': ('baseline', 'cosine'),
    # baseline "unsupervised"
    'cosine': ('threshold', 'cosine'),

    # baseline memorizations
    'lhs': ('linear', 'lhs'),
    'rhs': ('linear', 'rhs'),
    'concat': ('linear', 'concat'),
    'concat2': ('lr2', 'concat'),

    # asym models
    'diff': ('linear', 'diff'),
    'diffsq': ('linear', 'diffsq'),

    # asym lr
    'diff2': ('lr2', 'diff'),

    # asym lr
    'diffsq2': ('lr2', 'diffsq'),

    # rb models
    'diffrbf': ('rbf', 'diff'),
    'concatrbf': ('rbf', 'concat'),

    # others I dont want now
    #('lhs', 'lr1', 'lhs'),
    #('rhs', 'lr1', 'rhs'),
    #('concat', 'lr1', 'concat'),
    #('diff', 'lr1', 'diff'),
    #('diffsq', 'lr1', 'diffsq'),

    #('lhs', 'lr2', 'lhs'),
    #('rhs', 'lr2', 'rhs'),
    #('concat', 'lr2', 'concat'),
    #('diff', 'lr2', 'diff'),
    #('diffsq', 'lr2', 'diffsq'),

    #('diffpoly', 'poly2', 'diff'),
}

def words2matrix(dataseries, space):
    return np.array(list(dataseries.apply(lambda x: space[x])))

def generate_cosine_matrix(data, space):
    lhs = words2matrix(data.word1, space)
    rhs = words2matrix(data.word2, space)
    return np.array([np.sum(np.multiply(lhs, rhs), axis=1)]).T

def generate_diff_matrix(data, space):
    lhs = words2matrix(data.word1, space)
    rhs = words2matrix(data.word2, space)

    # difference vector
    diff = rhs - lhs

    return diff

def generate_diffsq_matrix(data, space):
    lhs = words2matrix(data.word1, space)
    rhs = words2matrix(data.word2, space)

    # difference vector
    diff = rhs - lhs
    # element wise squared diffs
    diff_sq = np.power(diff, 2)

    X = np.concatenate([diff, diff_sq], axis=1)
    return X

def generate_concat_matrix(data, space):
    lhs = words2matrix(data.word1, space)
    rhs = words2matrix(data.word2, space)

    X = np.concatenate([lhs, rhs], axis=1)
    return X

def generate_lhs_matrix(data, space):
    lhs = words2matrix(data.word2, space)
    return lhs

def generate_rhs_matrix(data, space):
    rhs = words2matrix(data.word2, space)
    return rhs

def generate_feature_matrix(data, space, features):
    if features == 'cosine':
        X = generate_cosine_matrix(data, space)
    elif features == 'lhs':
        X = generate_lhs_matrix(data, space)
    elif features == 'rhs':
        X = generate_rhs_matrix(data, space)
    elif features == 'concat':
        X = generate_concat_matrix(data, space)
    elif features == 'diff':
        X = generate_diff_matrix(data, space)
    elif features == 'diffsq':
        X = generate_diffsq_matrix(data, space)
    else:
        raise ValueError("Can't generate %s features" % features)
    y = data.entails.as_matrix()
    return X, y

def classifier_factory(name):
    if name == 'linear':
        return svm.LinearSVC()
    elif name == 'poly2':
        return svm.SVC(kernel='poly', degree=2)
    elif name == 'threshold':
        return ThresholdClassifier()
    elif name == 'rbf':
        return svm.SVC(kernel='rbf')
    elif name == 'lr2':
        return linear_model.LogisticRegression(penalty='l2', solver='liblinear')
    elif name == 'lr1':
        return linear_model.LogisticRegression(penalty='l1', solver='liblinear')
    elif name == 'baseline':
        return dummy.DummyClassifier(strategy='most_frequent')
    elif name == 'levy':
        # todo this
        return None
    else:
        raise ValueError("Don't know about %s models." % name)

def load_setup(setupname):
    kl, fe = SETUPS[setupname]
    return classifier_factory(kl), fe
