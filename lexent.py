#!/usr/bin/env python
import numpy as np
import pandas as pd
from collections import defaultdict
from functools import partial

from sklearn import svm, dummy, metrics
from sklearn import cross_validation
from scipy.stats import linregress

from space import load_mikolov_text
from threshold_classifier import ThresholdClassifier

np.random.seed(31337)

SPACES = [
    #'spaces/bow5.words',
    #'spaces/bow2.words',
    #'spaces/deps.words',
    'spaces/svd.bow5.words',
    'spaces/svd.bow2.words',
    'spaces/svd.deps.words',
]
#SPACE_FILENAME = "spaces/bow5.words"
#SPACE_FILENAME = "spaces/svd.bow2.words"
#SPACE_FILENAME = "spaces/svd.deps.words"

#DATA_FOLDER = "data/kotlerman2010"
#DATA_FOLDER = "data/bless2011"
#DATA_FOLDER = "data/baroni2012"
#DATA_FOLDER = "data/turney2014"
#DATA_FOLDER = "data/levy2014"

DATASETS = [
    #"data/kotlerman2010",
    "data/bless2011",
    "data/baroni2012",
    "data/turney2014",
    #"data/levy2014",
]


N_FOLDS = 20

def shorten(name):
    return name[name.rindex('/')+1:]

def generate_folds_all(data, n_folds):
    # get unique words
    words = list(set(data.word1).union(set(data.word2)))
    # randomize the list
    np.random.shuffle(words)
    # split into n_folds roughly equal groups
    folds_index = []
    for i in xrange(n_folds):
        folds_index.append(set(words[i::n_folds]))

    # so now we've got 10 folds, based on the lhs. now we need
    # to make sure that we don't have vocab overlap!
    folds = []
    for i in xrange(n_folds):
        idx = folds_index[i]
        testmask = data.word1.apply(lambda x: x in idx) | data.word2.apply(lambda x: x in idx)
        usedwords = set(data[testmask].word1).union(set(data[testmask].word2))
        trainmask = data.word1.apply(lambda x: x not in usedwords) & data.word2.apply(lambda x: x not in usedwords)

        #wasted = ~(trainmask | testmask)
        #print "wasted:", np.sum(wasted)/float(len(wasted))

        trainvocab = set(data[trainmask].word1).union(set(data[trainmask].word2))
        testvocab = set(data[testmask].word1).union(set(data[testmask].word2))
        assert len(trainvocab.intersection(testvocab)) == 0, trainvocab.intersection(testvocab)

        folds.append((data.index[trainmask], data.index[testmask]))

    return folds


def generate_folds_lhs(data, n_folds):
    # get unique words
    lhwords = list(set(data.word1))
    # randomize the list
    np.random.shuffle(lhwords)
    # split into n_folds roughly equal groups
    folds_index = []
    for i in xrange(n_folds):
        folds_index.append(set(lhwords[i::n_folds]))

    # so now we've got 10 folds, based on the lhs. now we need
    # to make sure that we don't have vocab overlap!
    folds = []
    for i in xrange(n_folds):
        idx = folds_index[i]
        testmask = data.word1.apply(lambda x: x in idx)
        rhswords = set(data.word2[testmask])
        # no words on rhs of test in rhs of train
        rhsmask = data.word2.apply(lambda x: x in rhswords)
        # no words in lhs of test in rhs of train
        rhsmask2 = data.word2.apply(lambda x: x in idx)
        # no words in rhs of test in lhs of train
        rhsmask3 = data.word1.apply(lambda x: x in rhswords)
        # overlap of these 3 conditions
        trainmask = ~(testmask | rhsmask | rhsmask2 | rhsmask3)

        #wasted = ~(trainmask | testmask)
        #print "wasted:", np.sum(wasted)/float(len(wasted))

        trainvocab = set(data[trainmask].word1).union(set(data[trainmask].word2))
        testvocab = set(data[testmask].word1).union(set(data[testmask].word2))
        assert len(trainvocab.intersection(testvocab)) == 0

        folds.append((data.index[trainmask], data.index[testmask]))

    return folds

def generate_pseudo_data(data, test_fold, percent_fake=0.5):
    t = data.ix[test_fold]
    tp = t[t.entails == True]
    tp1 = list(set(tp.word1))
    tp2 = list(set(tp.word2))
    tps = set(zip(tp.word1, tp.word2))

    fps = set()
    while len(fps) < len(tps) / (2 * percent_fake):
        i = np.random.randint(len(tp1))
        w1 = tp1[i]
        j = np.random.randint(len(tp2))
        w2 = tp2[j]

        if (w1, w2) in tps:
            continue
        else:
            fps.add((w1, w2))

    return pd.DataFrame(
        [{'word1': w1, 'word2': w2, 'entails': True}
            for w1, w2 in tps] +
        [{'word1': w1, 'word2': w2, 'entails': False}
            for w1, w2 in fps]
    )





def words2matrix(dataseries, space):
    return np.array(list(dataseries.apply(lambda x: space[x.lower()])))

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

def generate_model():
    return ThresholdClassifier()
    #return dummy.DummyClassifier()
    #return svm.LinearSVC()

def model_factory(name):
    if name == 'linear':
        return svm.LinearSVC()
    elif name == 'poly2':
        return svm.SVC(kernel='poly', degree=2)
    elif name == 'threshold':
        return ThresholdClassifier()
    elif name == 'rbf':
        return svm.SVC(kernel='rbf')
    elif name == 'levy':
        # todo this
        return None
    else:
        raise ValueError("Don't know about %s models." % name)


if __name__ == '__main__':

    setups = [
        ('cosine', 'threshold', 'cosine'),
        #('lhs', 'linear', 'lhs'),
        #('rhs', 'linear', 'rhs'),
        ('concat', 'linear', 'concat'),
        ('diff', 'linear', 'diff'),
        #('diffpoly', 'poly2', 'diff'),
        #('diffrbf', 'rbf', 'diff'),
        ('diffsq', 'linear', 'diffsq'),
    ]


    me_results_csv = []

    for DATA_FOLDER in DATASETS:
        data = pd.read_table("%s/data.tsv" % DATA_FOLDER, header=None, names=('word1', 'word2', 'entails'))
        DATA_FOLDER_SHORT = shorten(DATA_FOLDER)
        for SPACE_FILENAME in SPACES:
            SPACE_FILENAME_SHORT = shorten(SPACE_FILENAME)
            print "SPACE: %s" % SPACE_FILENAME_SHORT
            print "DATA: %s" % DATA_FOLDER_SHORT
            space = load_mikolov_text(SPACE_FILENAME).normalize()

            # filter for vocabulary
            mask1 = data.word1.apply(lambda x: x in space)
            mask2 = data.word2.apply(lambda x: x in space)
            data = data[mask1 & mask2].reset_index(drop=True)

            # need our folds for cross validation
            folds = generate_folds_lhs(data, n_folds=N_FOLDS)

            print "Train sizes:", np.array([len(f[0]) for f in folds])
            print "Test sizes: ", np.array([len(f[1]) for f in folds])

            print "Standard Classification:"
            for name, model_name, features in setups:
                print "Name: %s (model: %s, features: %s)" % (name, model_name, features)

                # generate features
                X, y = generate_feature_matrix(data, space, features)

                # load up the model
                model = model_factory(model_name)

                # perform cross validation
                scores = cross_validation.cross_val_score(model, X, y, scoring='f1', cv=folds)#, n_jobs=-1)
                #scores = cross_validation.cross_val_score(model, X, y, cv=folds, n_jobs=-1)
                stderr = np.std(scores)/np.sqrt(N_FOLDS)
                mean = np.mean(scores)

                #print "F1: %.3f" % mean
                #print ">=: %.3f" % (mean - stderr)
                #print "<=: %.3f" % (mean + stderr)
                print "    F1:  %.3f <=   %.3f  <= %.3f" % (mean - stderr, mean, mean + stderr)
                # print " ".join("%.3f" % s for s in scores)
                print

            print "False Positive Issue:"
            recalls = defaultdict(list)
            match_errs = defaultdict(list)
            for fold in folds:
                fake_data = generate_pseudo_data(data, fold[1], 0.5)
                for name, model_name, features in setups:
                    Xtr, ytr = generate_feature_matrix(data.ix[fold[0]], space, features)
                    Xte, yte = generate_feature_matrix(fake_data, space, features)

                    model = model_factory(model_name)
                    model.fit(Xtr, ytr)

                    preds = model.predict(Xte)
                    recalls[name].append(metrics.recall_score(yte, preds))
                    #match_errs[name].append(metrics.recall_score(~yte, preds))
                    match_errs[name].append(float(np.sum(~yte & preds)) / np.sum(~yte))

            print "%-10s   r        m         b      recall   materr" % (" ")
            for item in recalls.keys():
                R = np.mean(recalls[item])
                ME = np.mean(match_errs[item])
                #for r, me in zip(recalls[item], match_errs[item], trues[item]):
                #    print "%-10s %.3f   %.3f   %.3f" % (item, r, me, t)
                m, b, r, p, se = linregress(recalls[item], match_errs[item])
                print "%-10s  %6.3f   %6.3f    %6.3f  %.3f    %.3f" % (item, r, m, b, R, ME)

                for R, ME in zip(recalls[item], match_errs[item]):
                    me_results_csv.append({
                        'name': item,
                        'recall': R,
                        'match_error': ME,
                        'space': SPACE_FILENAME_SHORT,
                        'data': DATA_FOLDER_SHORT,
                    })

    pd.DataFrame(me_results_csv).to_csv("matcherror.csv", index=False)






