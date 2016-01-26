#!/usr/bin/env python

import numpy as np

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


