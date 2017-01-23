#!/usr/bin/env python

import numpy as np

def generate_folds_random(rng, data, n_folds):
    assignments = rng.randint(n_folds, size=len(data))
    folds = []
    for i in xrange(n_folds):
        testmask = (assignments == i)
        valmask = (assignments == (i + 1) % n_folds)
        trainmask = ~(testmask | valmask)
        folds.append((np.array(data.index[trainmask]),
                      np.array(data.index[valmask]),
                      np.array(data.index[testmask])))
    return folds

def generate_folds_levy(data, data_name):
    levy_train = pd.read_table("%s/data_lex_train.tsv" % DATA_FOLDER, header=None, names=('word1', 'word2', 'entails'))
    levy_val = pd.read_table("%s/data_lex_val.tsv" % DATA_FOLDER, header=None, names=('word1', 'word2', 'entails'))
    levy_test = pd.read_table("%s/data_lex_test.tsv" % DATA_FOLDER, header=None, names=('word1', 'word2', 'entails'))

    data_copy = data.copy()
    data_copy['idx'] = data_copy.index

    train_merge = pd.merge(data_copy, levy_train, on=('word1', 'word2', 'entails'))
    val_merge = pd.merge(data_copy, levy_val, on=('word1', 'word2', 'entails'))
    test_merge = pd.merge(data_copy, levy_test, on=('word1', 'word2', 'entails'))

    return [(np.array(train_merge.idx), np.array(val_merge.idx), np.array(test_merge.idx))]

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
        vidx = folds_index[(i + 1) % n_folds]
        testmask = data.word1.apply(lambda x: x in idx)
        valmask = data.word1.apply(lambda x: x in vidx)

        testrhswords = set(data.word2[testmask])
        # no words on rhs of test in rhs of train
        testrhsmask1 = data.word2.apply(lambda x: x in testrhswords)
        # no words in lhs of test in rhs of train
        testrhsmask2 = data.word2.apply(lambda x: x in idx)
        # no words in rhs of test in lhs of train
        testrhsmask3 = data.word1.apply(lambda x: x in testrhswords)

        valmask = valmask & ~(testmask | testrhsmask1 | testrhsmask2 | testrhsmask3)

        valrhswords = set(data.word2[valmask])
        valrhsmask1 = data.word2.apply(lambda x: x in valrhswords)
        valrhsmask2 = data.word2.apply(lambda x: x in vidx)
        valrhsmask3 = data.word1.apply(lambda x: x in valrhswords)

        trainmask = ~(testmask | testrhsmask1 | testrhsmask2 | testrhsmask3 |
                      valmask  | valrhsmask1  | valrhsmask2  | valrhsmask3)

        #wasted = ~(trainmask | testmask)
        #print "wasted:", np.sum(wasted)/float(len(wasted))

        trainvocab = set(data[trainmask].word1).union(set(data[trainmask].word2))
        valvocab = set(data[valmask].word1).union(set(data[valmask].word2))
        testvocab = set(data[testmask].word1).union(set(data[testmask].word2))
        #assert len(trainvocab.intersection(valvocab)) == 0
        assert len(trainvocab.intersection(testvocab)) == 0
        assert len(valvocab.intersection(testvocab)) == 0

        folds.append((np.array(data.index[trainmask]), np.array(data.index[valmask]), np.array(data.index[testmask])))

    all_seen = set()
    for train, val, test in folds:
        all_seen.update(test)

    assert len(all_seen) == len(data)

    return folds


