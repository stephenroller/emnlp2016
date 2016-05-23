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

def generate_folds_all(rng, data, percent_test, num_trials):
    all_vocab = list(set(data.word1).union(data.word2))

    folds = []
    for i in xrange(num_trials):
        rng.shuffle(all_vocab)
        V = len(all_vocab)
        S = int(V*percent_test)
        test_vocab = set(all_vocab[:S])
        train_vocab = set(all_vocab[S:])

        testmask1 = data.word1.apply(lambda x: x in test_vocab)
        testmask2 = data.word2.apply(lambda x: x in test_vocab)
        testmask = testmask1 & testmask2

        trainmask1 = data.word1.apply(lambda x: x in train_vocab)
        trainmask2 = data.word2.apply(lambda x: x in train_vocab)
        trainmask = trainmask1 & trainmask2

        test_vocab2 = set(data[testmask].word1).union(set(data[testmask].word2))
        train_vocab2 = set(data[trainmask].word1).union(set(data[trainmask].word2))
        assert len(train_vocab2.intersection(test_vocab2)) == 0

        folds.append((np.array(data.index[trainmask]), np.array(data.index[testmask])))

    return folds

def generate_folds_column(rng, data, column):
    stratwords = list(set(data[column]))

    folds = []

    for sw in stratwords:
        testmask = (data[column] == sw)
        test = data[testmask]
        left_set = set(test['word1'])
        right_set = set(test['word2'])
        test_words = left_set.union(right_set)
        trainmask = (~data['word1'].apply(test_words.__contains__) &
                     ~data['word2'].apply(test_words.__contains__))
        folds.append((np.array(data.index[trainmask]), np.array(data.index[testmask])))

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
        #trainmask = ~(testmask | rhsmask | rhsmask2 | rhsmask3)
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


