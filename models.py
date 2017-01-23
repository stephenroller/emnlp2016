#!/usr/bin/env python

import numpy as np
from sklearn import svm, linear_model, dummy

from custom_classifiers import ThresholdClassifier, ksim_kernel

SETUPS = {
    # baseline most common
    'baseline': ('baseline', 'cosine'),
    # baseline "unsupervised"
    'cosine': ('threshold', 'cosine'),

    # baseline memorizations
    'lhs': ('lr2', 'lhs'),
    'rhs': ('lr2', 'rhs'),
    'concat': ('linear', 'concat'),
    'concat2': ('lr2', 'concat'),

    # asym models
    'diff': ('linear', 'diff'),
    'diffsq': ('linear', 'diffsq'),

    # asym lr
    'diff1': ('lr1', 'diff'),
    'diff2': ('lr2', 'diff'),

    # asym lr
    'diffsq1': ('lr1', 'diffsq'),
    'diffsq2': ('lr2', 'diffsq'),

    # rb models
    'diffrbf': ('rbf', 'diff'),
    'concatrbf': ('rbf', 'concat'),
    'asymrbf': ('rbf', 'diffsq'),

    # other models
    'bdsm': ('bdsm', 'concat'),
    'stephen': ('stephen', 'concat'),
    'ksim': ('ksim', 'concat'),
    'sigmoid': ('sigmoid', 'concat'),

    'concat+asym': ('lr2', 'concat+asym'),
    'concat+diff': ('lr2', 'concat+diff'),
    'concat+sq': ('lr2', 'concat+sq'),
    'concat+cos': ('lr2', 'concat+cos'),
    'concat+cosrbf': ('rbf', 'concat+cos'),

    'super': ('super', 'concat'),
    'super-cos': ('super-cos', 'concat'),
    'super-incl': ('super-incl', 'concat'),
    'super-proj': ('super-proj', 'concat'),

    'sq': ('linear', 'sq'),

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

    'poly': ('poly', 'concat'),
}

def _lsplit(needle, haystack):
    i = haystack.index(needle)
    return haystack[:i], haystack[i+1:]

def topk(matrix, k=10):
    return np.argsort(-matrix, 1)[:,:k]

def find_indices(space, relation, candidates):
    tofind = (relation + '+' + space.vocab[c] for c in candidates)
    for f in tofind:
        i = space.clookup.get(f)
        if i:
            return i
    return 0

def generate_relation_matrix(data, space):
    retval_l = []
    retval_r = []
    from collections import Counter
    relations = Counter([_lsplit('+', c)[0] for c in space.cvocab[1:]])
    relations = [k for k, v in relations.iteritems() if v >= 10]
    setattr(space, 'relations', relations)

    ## first we gotta get the nearest neighbors
    #allvocab = set(list(data['word1']) + list(data['word2']))
    #allvecs = space.matrix[[space.lookup[v] for v in allvocab]]
    #nearest_neighbors = topk(allvecs.dot(space.matrix.T))
    #nn_lookup = {v : i for i, v in enumerate(allvocab)}
    #import ipdb; ipdb.set_trace()
    #print nearest_neighbors
    
    k = 0
    for i, row in data.iterrows():
        k = k + 1
        #print "%d/%d" % (k , len(data))
        word1 = row['word1']
        word2 = row['word2']
        lhs = space.matrix[space.lookup[word1]]
        rhs = space.matrix[space.lookup[word2]]
        #indices = [find_indices(space, r, nearest_neighbors[nn_lookup[word2]]) for r in relations]
        indices = [find_indices(space, r, [space.lookup[word2]]) for r in relations]
        ests_l = space.cmatrix[indices].dot(lhs)
        tofind = [(r + '+' + word1) for r in relations]
        #indices = [find_indices(space, r, nearest_neighbors[nn_lookup[word2]]) for r in relations]
        indices = [find_indices(space, r, [space.lookup[word1]]) for r in relations]
        ests_r = space.cmatrix[indices].dot(rhs)
        retval_l.append(ests_l)
        retval_r.append(ests_r)
    retval = np.concatenate([np.array(retval_l), np.array(retval_r)], axis=1)
    #retval = retval.clip(0, 20)
    retval = np.exp(retval)
    from sklearn.preprocessing import normalize
    return normalize(retval, norm='l2')
    return retval


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
    lhs = words2matrix(data.word1, space)
    return lhs

def generate_rhs_matrix(data, space):
    rhs = words2matrix(data.word2, space)
    return rhs

def bin(X, nbins=20):
    equa0 = (X <= 0.0)
    equa1 = (X >= 1.0)

    binned = []

    groups = range(nbins+1)
    for g1, g2 in zip(groups, groups[1:]):
        l, u = g1/float(nbins), g2/float(nbins)
        binned.append((l < X) & (X <= u))
    retval = np.concatenate([equa0, equa1] + binned, axis=1)
    return retval

def generate_feature_matrix(data, space, features):
    if features == 'relation':
        X = generate_relation_matrix(data, space)
    elif features == 'cosine':
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
    elif features == 'concat+cos':
        X1 = generate_lhs_matrix(data, space)
        X2 = generate_rhs_matrix(data, space)
        X3 = generate_cosine_matrix(data, space)
        X = np.concatenate([X1, X2, X3, bin(X3)], axis=1)
    elif features == 'sq':
        X = np.square(generate_diff_matrix(data, space))
    elif features == 'concat+diff':
        X1 = generate_diff_matrix(data, space)
        X2 = generate_lhs_matrix(data, space)
        X3 = generate_rhs_matrix(data, space)
        X = np.concatenate([X1, X2, X3], axis=1)
    elif features == 'concat+asym':
        X1 = generate_diff_matrix(data, space)
        X2 = generate_lhs_matrix(data, space)
        X3 = generate_rhs_matrix(data, space)
        X4 = np.square(X1)
        X = np.concatenate([X1, X2, X3, X4], axis=1)
    elif features == 'concat+sq':
        X1 = np.square(generate_diff_matrix(data, space))
        X2 = generate_lhs_matrix(data, space)
        X3 = generate_rhs_matrix(data, space)
        X = np.concatenate([X1, X2, X3], axis=1)
    elif features == 'diffrhs':
        X1 = generate_diffsq_matrix(data, space)
        X2 = generate_rhs_matrix(data, space)
        X = np.concatenate([X1, X2], axis=1)
    else:
        raise ValueError("Can't generate %s features" % features)
    y = data.label.as_matrix()
    return X, y

def dict_union(a, b):
    c = {}
    for k, v in a.iteritems():
        c[k] = v
    for k, v in b.iteritems():
        c[k] = v
    return c

def classifier_factory(name):
    #Cs = {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5], 'class_weight': ['balanced']}
    Cs = {'C': [1e+0, 1e-1, 1e+1, 1e-2, 1e+2, 1e-3, 1e+3, 1e-4, 1e+4], 'class_weight': ['balanced']}
    if name == 'linear':
        return svm.LinearSVC(dual=False), Cs
    elif name == 'ballinear':
        return svm.LinearSVC(class_weight='balanced', dual=False), Cs
    elif name == 'poly':
        return svm.SVC(kernel='poly', degree=3, shrinking=False, cache_size=8192, max_iter=2000), Cs
    elif name == 'threshold':
        return ThresholdClassifier(), {}
    elif name == 'rbf':
        return svm.SVC(kernel='rbf', class_weight='balanced', cache_size=8192, shrinking=False, max_iter=2000, C=1e+2), Cs
    elif name == 'lr2':
        return linear_model.LogisticRegression(penalty='l2', solver='liblinear'), Cs
    elif name == 'lr1':
        return linear_model.LogisticRegression(penalty='l1', solver='liblinear'), Cs
    elif name == 'baseline':
        return dummy.DummyClassifier(strategy='most_frequent'), {}
    elif name.startswith('super'):
        from custom_classifiers import SuperTreeClassifier
        remove = name[6:]
        extra = {}
        if remove:
            extra[remove] = False
        return SuperTreeClassifier(**extra), dict_union(Cs, {'n_features': [1, 2, 3, 4, 5, 6]})
    elif name == 'ksim':
        return svm.SVC(kernel=ksim_kernel, cache_size=8192, shrinking=False, max_iter=2000), Cs
    else:
        raise ValueError("Don't know about %s models." % name)

def load_setup(setupname):
    kl, fe = SETUPS[setupname]
    cl, hyper = classifier_factory(kl)
    return cl, fe, hyper
