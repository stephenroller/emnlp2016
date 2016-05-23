#!/usr/bin/env python

import numpy as np
from sklearn import svm, linear_model, dummy

from custom_classifiers import ThresholdClassifier, levy_kernel

SETUPS = {
    # baseline most common
    'baseline': ('baseline', 'cosine'),
    # baseline "unsupervised"
    'cosine': ('threshold', 'cosine'),

    # baseline memorizations
    'lhs': ('lr2', 'lhs'),
    'rhs': ('lr2', 'rhs'),
    'concat': ('linear', 'concat'),
    'balconcat': ('ballinear', 'concat'),
    'concat2': ('lr2', 'concat'),
    'balconcat2': ('ballr2', 'concat'),

    # asym models
    'diff': ('linear', 'diff'),
    'diffsq': ('linear', 'diffsq'),
    'baldiff': ('ballinear', 'diff'),
    'baldiffsq': ('ballinear', 'diffsq'),

    # asym lr
    'baldiff1': ('ballr1', 'diff'),
    'baldiff2': ('ballr2', 'diff'),
    'diff1': ('lr1', 'diff'),
    'diff2': ('lr2', 'diff'),

    # asym lr
    'baldiffsq1': ('ballr1', 'diffsq'),
    'baldiffsq2': ('ballr2', 'diffsq'),
    'diffsq1': ('lr1', 'diffsq'),
    'diffsq2': ('lr2', 'diffsq'),

    # rb models
    'diffrbf': ('rbf', 'diff'),
    'concatrbf': ('rbf', 'concat'),
    'asymrbf': ('rbf', 'diffsq'),

    # other models
    'bdsm': ('bdsm', 'concat'),
    'stephen': ('stephen', 'concat'),
    'levy': ('levy', 'concat'),
    'sigmoid': ('sigmoid', 'concat'),

    'balsqconcat': ('ballr2', 'sqconcat'),
    'balcoslhsrhs': ('ballr2', 'coslhsrhs'),
    'sqconcat': ('linear', 'sqconcat'),
    'coslhsrhs': ('linear', 'coslhsrhs'),

    'super': ('super', 'concat'),
    'super1': ('super1', 'concat'),
    'super2': ('super2', 'concat'),
    'super3': ('super3', 'concat'),
    'super4': ('super4', 'concat'),
    'super5': ('super5', 'concat'),
    'super6': ('super8', 'concat'),
    'super7': ('super7', 'concat'),
    'super8': ('super8', 'concat'),
    'super9': ('super9', 'concat'),
    'super10': ('super10', 'concat'),

    'sq': ('linear', 'sq'),
    'balsq': ('ballinear', 'sq'),

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

    'balpoly3': ('poly3', 'concat'),
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
    elif features == 'coslhsrhs':
        X1 = generate_lhs_matrix(data, space)
        X2 = generate_rhs_matrix(data, space)
        X3 = generate_cosine_matrix(data, space)
        X = np.concatenate([X1, X2, X3, bin(X3)], axis=1)
    elif features == 'sq':
        X = np.square(generate_diff_matrix(data, space))
    elif features == 'sqconcat':
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
    y = data.entails.as_matrix()
    return X, y

def classifier_factory(name):
    if name == 'linear':
        return svm.LinearSVC(dual=False)
    elif name == 'ballinear':
        return svm.LinearSVC(class_weight='balanced', dual=False)
    elif name == 'poly2':
        return svm.SVC(kernel='poly', degree=2, class_weight='balanced', shrinking=False, cache_size=8192, max_iter=2000)
    elif name == 'poly3':
        return svm.SVC(kernel='poly', degree=3, class_weight='balanced', shrinking=False, cache_size=8192, max_iter=2000)
    elif name == 'threshold':
        return ThresholdClassifier()
    elif name == 'sigmoid':
        return svm.SVC(kernel='sigmoid', class_weight='balanced', cache_size=8192, shrinking=False, max_iter=2000)
    elif name == 'rbf':
        return svm.SVC(kernel='rbf', class_weight='balanced', cache_size=8192, shrinking=False, max_iter=2000, C=1e+2)
    elif name == 'lr2':
        return linear_model.LogisticRegression(penalty='l2', solver='liblinear')
    elif name == 'lr1':
        return linear_model.LogisticRegression(penalty='l1', solver='liblinear')
    elif name == 'ballr2':
        return linear_model.LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced')
    elif name == 'ballr1':
        return linear_model.LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
    elif name == 'baseline':
        return dummy.DummyClassifier(strategy='most_frequent')
    elif name == 'stephen':
        from nn import Stephen
        return Stephen()
    elif name == 'bdsm':
        from nn import BDSM
        return BDSM(5)
    elif name == 'tree':
        import sklearn.tree
        return sklearn.tree.DecisionTreeClassifier()
    elif name.startswith('super'):
        from custom_classifiers import SuperTreeClassifier
        if name == 'super': n = 4
        else: n = int(name[5:])
        return SuperTreeClassifier(n_features=n)
    elif name == 'levy':
        return svm.SVC(kernel=levy_kernel, cache_size=8192, class_weight='balanced', shrinking=False, max_iter=5000)
    else:
        raise ValueError("Don't know about %s models." % name)

def load_setup(setupname):
    kl, fe = SETUPS[setupname]
    return classifier_factory(kl), fe
