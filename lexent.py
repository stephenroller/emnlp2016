#!/usr/bin/env python
import os
import os.path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from collections import defaultdict
from functools import partial

from sklearn import svm, linear_model
from sklearn import cross_validation, metrics
from scipy.stats import linregress

from custom_classifiers import ThresholdClassifier

from utdeftvs import load_numpy


SPACES_DIR = "/scratch/cluster/roller/spaces/giga+bnc+uk+wiki2015/output"
#SPACES_DIR = '/scratch/cluster/roller/spaces/giga+bnc+uk+wiki2015/dependency/output/foo'
SPACES = [
    #'spaces/bow5.words',
    #'spaces/bow2.words',
    #'spaces/deps.words',
    #'spaces/svd.bow5.words',
    #'spaces/svd.bow2.words',
    #'spaces/svd.deps.words',
    #'spaces/window2.vectorspace.ppmi.svd_300.words',
    #'spaces/sentence.vectorspace.ppmi.svd_300.words',
]
#SPACES = SPACES + [s for s in os.listdir(SPACES_DIR) if ('.npz' in s) and ('.svd300' in s)]
SPACES = ['dependency.svd300.ppmi.250k.1m.npz', 'window2.svd300.ppmi.top250k.top20k.npz']

DATASETS = [
    #"data/kotlerman2010",
    "data/bless2011",
    "data/baroni2012",
    "data/turney2014",
    "data/levy2014",
    #"data/eval_hyper",
    #"data/eval_entail",
]


N_FOLDS = 20

def shorten(name):
    return name[name.rindex('/')+1:]

def generate_folds_levy(data, data_name):
    levy_train = pd.read_table("%s/data_lex_train.tsv" % DATA_FOLDER, header=None, names=('word1', 'word2', 'entails'))
    levy_test = pd.read_table("%s/data_lex_test.tsv" % DATA_FOLDER, header=None, names=('word1', 'word2', 'entails'))

    data_copy = data.copy()
    data_copy['idx'] = data_copy.index

    train_merge = pd.merge(data_copy, levy_train, on=('word1', 'word2', 'entails'))
    test_merge = pd.merge(data_copy, levy_test, on=('word1', 'word2', 'entails'))

    return [(np.array(train_merge.idx), np.array(test_merge.idx))]

def generate_folds_all(data, n_folds):
    all_vocab = list(set(data.word1).union(data.word2))
    np.random.shuffle(all_vocab)

    folds_index = []
    for i in xrange(n_folds):
        folds_index.append(set(all_vocab[i::n_folds]))

    folds = []
    for i in xrange(n_folds):
        idx = folds_index[i]
        testmask = data.word1.apply(lambda x: x in idx) & data.word2.apply(lambda x: x in idx)
        trainmask = data.word1.apply(lambda x: x not in idx) & data.word2.apply(lambda x: x not in idx)

        trainvocab = set(data[trainmask].word1).union(set(data[trainmask].word2))
        testvocab = set(data[testmask].word1).union(set(data[testmask].word2))
        assert len(trainvocab.intersection(testvocab)) == 0

        folds.append((np.array(data.index[trainmask]), np.array(data.index[testmask])))

    #all_seen = set()
    #for train, test in folds:
    #    all_seen.update(test)
    #assert len(all_seen) == len(data)

    return folds






def generate_folds_lhs(rng, data, n_folds):
    # get unique words
    lhwords = list(set(data.word1))
    # randomize the list
    rng.shuffle(lhwords)
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

        folds.append((np.array(data.index[trainmask]), np.array(data.index[testmask])))

    all_seen = set()
    for train, test in folds:
        all_seen.update(test)

    assert len(all_seen) == len(data)

    return folds

#def generate_pseudo_data(data, test_fold, percent_fake=0.5):
#    t = data.ix[test_fold]
#    tp = t[t.entails == True]
#    tp1 = list(set(tp.word1))
#    tp2 = list(set(tp.word2))
#    tps = set(zip(tp.word1, tp.word2))
#
#    fps = set()
#    fails = 0
#    while len(fps) < len(tps) / (2 * percent_fake):
#        i = np.random.randint(len(tp1))
#        w1 = tp1[i]
#        j = np.random.randint(len(tp2))
#        w2 = tp2[j]
#
#        if (w1, w2) in tps:
#            fails += 1
#            if fails >= 1000:
#                break
#            continue
#        else:
#            fps.add((w1, w2))
#
#    return pd.DataFrame(
#        [{'word1': w1, 'word2': w2, 'entails': True}
#            for w1, w2 in tps] +
#        [{'word1': w1, 'word2': w2, 'entails': False}
#            for w1, w2 in fps]
#    )

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

def generate_feature_matrix(data, space, features, global_vocab):
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

def model_factory(name):
    if name == 'linear':
        return svm.LinearSVC()
    elif name == 'poly2':
        return svm.SVC(kernel='poly', degree=2)
    elif name == 'threshold':
        return ThresholdClassifier()
    elif name == 'rbf':
        return svm.SVC(kernel='rbf')
    elif name == 'lr2':
        return linear_model.LogisticRegression(penalty='l2')
    elif name == 'lr1':
        return linear_model.LogisticRegression(penalty='l1')
    elif name == 'levy':
        # todo this
        return None
    else:
        raise ValueError("Don't know about %s models." % name)

def perform_trial(seed, setups, global_vocab):
    output = []
    rng = np.random.RandomState(seed)
    for DATA_FOLDER in DATASETS:
        data = pd.read_table("%s/data.tsv" % DATA_FOLDER, header=None, names=('word1', 'word2', 'entails'))
        data['word1'] = data['word1'].apply(lambda x: x.lower() + '/NN')
        data['word2'] = data['word2'].apply(lambda x: x.lower() + '/NN')


        mask1 = data.word1.apply(lambda x: x in global_vocab)
        mask2 = data.word2.apply(lambda x: x in global_vocab)

        data = data[mask1 & mask2].reset_index(drop=True)

        # need our folds for cross validation
        folds = generate_folds_lhs(rng, data, n_folds=N_FOLDS)
        #folds = generate_folds_levy(data, DATA_FOLDER)
        DATA_FOLDER_SHORT = shorten(DATA_FOLDER)

        for SPACE_FILENAME in SPACES:
            SPACE_FILENAME_SHORT = SPACE_FILENAME
            output += ["SPACE: %s" % SPACE_FILENAME_SHORT]
            output += ["DATA: %s" % DATA_FOLDER_SHORT]
            space = load_numpy(os.path.join(SPACES_DIR, SPACE_FILENAME)).normalize()

            train_sizes = np.array([len(f[0]) for f in folds])
            test_sizes = np.array([len(f[1]) for f in folds])
            output += ["training set sizes: %f" % np.mean(train_sizes)]
            output += [" testing set sizes: %f" % np.mean(test_sizes)]
            output += ["test as %% of train: %f" % np.mean(np.divide(0.0 + test_sizes, train_sizes + test_sizes))]

            output += ["Standard Classification:"]
            results = []
            for name, model_name, features in setups:
                # load up the model
                model = model_factory(model_name)

                data_with_predictions = data.copy()
                data_with_predictions['prediction'] = False
                data_with_predictions['fold'] = -1

                # perform cross validation
                scores = []
                for foldno, (train, test) in enumerate(folds):
                    # generate features
                    train_X, train_y = generate_feature_matrix(data.ix[train], space, features, global_vocab)
                    test_X, test_y = generate_feature_matrix(data.ix[test], space, features, global_vocab)

                    #print train_X.shape, test_X.shape

                    #train_X, train_y = X[train], y[train]
                    #test_X, test_y = X[test], y[test]
                    model.fit(train_X, train_y)
                    preds_y = model.predict(test_X)
                    data_with_predictions.loc[test,'prediction'] = preds_y
                    data_with_predictions.loc[test,'fold'] = foldno
                    scores.append(metrics.f1_score(test_y, preds_y))

                stderr = np.std(scores)/np.sqrt(N_FOLDS)
                mean = np.mean(scores)

                #f1_results_csv.append({
                #    'data': DATA_FOLDER_SHORT,
                #    'space': SPACE_FILENAME_SHORT,
                #    'model': model_name,
                #    'features': features,
                #    'dims': space.matrix.shape[1],
                #    'mean': mean,
                #    'std': stderr,
                #    'n_folds': N_FOLDS,
                #    'seed': seed,
                #})

                # make sure we've seen every item
                assert len(data_with_predictions[data_with_predictions['fold'] == -1]) == 0

                #data_with_predictions.to_csv("results/data:%s_space:%s_modelname:%s_features:%s.csv" %
                #    (DATA_FOLDER_SHORT, SPACE_FILENAME_SHORT, model_name, features), index=False)

                #print "F1: %.3f" % mean
                #print ">=: %.3f" % (mean - stderr)
                #print "<=: %.3f" % (mean + stderr)
                results.append((mean, (model_name, features, mean - stderr, mean, mean + stderr)))
            results = sorted(results)
            for val, item in results:
                output += ["    %-10s %-10s   F1:  %.3f <=   %.3f  <= %.3f" % item]

            #print "False Positive Issue:"
            #recalls = defaultdict(list)
            #match_errs = defaultdict(list)
            #for fold in folds:
            #    fake_data = generate_pseudo_data(data, fold[1], 0.5)
            #    if len(fake_data) == 0:
            #        continue
            #    for name, model_name, features in setups:
            #        Xtr, ytr = generate_feature_matrix(data.ix[fold[0]], space, features, global_vocab)
            #        Xte, yte = generate_feature_matrix(fake_data, space, features, global_vocab)

            #        model = model_factory(model_name)
            #        model.fit(Xtr, ytr)

            #        preds = model.predict(Xte)
            #        recalls[(name, model_name, features)].append(metrics.recall_score(yte, preds))
            #        #match_errs[name].append(metrics.recall_score(~yte, preds))
            #        match_errs[(name, model_name, features)].append(float(np.sum(~yte & preds)) / np.sum(~yte))

            #print "%-10s  %-10s   r        m         b      recall   materr" % (" ", " ")
            #for item in recalls.keys():
            #    name, model_name, features = item

            #    R = np.mean(recalls[item])
            #    ME = np.mean(match_errs[item])
            #    #for r, me in zip(recalls[item], match_errs[item], trues[item]):
            #    #    print "%-10s %.3f   %.3f   %.3f" % (item, r, me, t)
            #    m, b, r, p, se = linregress(recalls[item], match_errs[item])
            #    print "%-10s  %-10s  %6.3f   %6.3f    %6.3f  %.3f    %.3f" % (model_name, features, r, m, b, R, ME)

            #    for R, ME in zip(recalls[item], match_errs[item]):
            #        me_results_csv.append({
            #            'name': name,
            #            'model': model_name,
            #            'features': features,
            #            'Features + Data': features + " " + DATA_FOLDER_SHORT,
            #            'recall': R,
            #            'match_error': ME,
            #            'space': SPACE_FILENAME_SHORT,
            #            'data': DATA_FOLDER_SHORT,
            #            'seed': seed,
            #        })
    return output




if __name__ == '__main__':

    setups = [
        ('cosine', 'threshold', 'cosine'),

        #('lhs', 'linear', 'lhs'),
        #('rhs', 'linear', 'rhs'),
        ('concat', 'linear', 'concat'),
        ('diff', 'linear', 'diff'),
        ('diffsq', 'linear', 'diffsq'),

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
        ('diffrbf', 'rbf', 'diff'),
        ('concatrbf', 'rbf', 'concat'),
    ]


    f1_results_csv = []
    me_results_csv = []

    global_vocab = None
    for SPACE_FILENAME in SPACES:
        space = load_numpy(os.path.join(SPACES_DIR, SPACE_FILENAME))
        if not global_vocab:
            global_vocab = set(space.vocab)
        else:
            global_vocab = global_vocab.intersection(set(space.vocab))

        results = Parallel(n_jobs=20)(
            delayed(perform_trial)(seed, setups, global_vocab)
            for seed in xrange(1, 21))

        for r in results:
            print "\n".join(r)

    #pd.DataFrame(me_results_csv).to_csv("matcherror.csv", index=False)
    #pd.DataFrame(f1_results_csv).to_csv("allresults.csv", index=False)

