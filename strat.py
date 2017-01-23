#!/usr/bin/env python

import os
import os.path
import argparse
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.stats import linregress

from utdeftvs import load_numpy

import fold
import models

from custom_classifiers import ThresholdClassifier
tc = ThresholdClassifier()

N_FOLDS = 20

# before anything else, configure the logger
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s   %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def longest_vector_posmatch(search, space):
    if not hasattr(space, 'posless'):
        setattr(space, 'posless', defaultdict(dict))
        for i, v in enumerate(space.vocab):
            sep = v.rindex('/')
            word, pos = v[:sep], v[sep+1:]
            space.posless[word][pos] = space.magn[i]
    if search not in space.posless:
        return None
    options = space.posless[search]
    pos = max(options.keys(), key=options.__getitem__)
    #print search, options, pos
    #import ipdb; ipdb.set_trace()
    return search + '/' + pos

def always_nn_posmatch(search, space):
    return search + '/NN'

best_pos_match = always_nn_posmatch
#best_pos_match = longest_vector_posmatch

def main():
    parser = argparse.ArgumentParser('Lexical Entailment Classifier')
    parser.add_argument('--data', '-d', help='Input file')
    parser.add_argument('--space', '-s', help='Distributional space')
    parser.add_argument('--model', '-m', help='Model setup', choices=models.SETUPS.keys())
    args = parser.parse_args()

    logger.debug('Lexent Arguments: ')
    logger.debug(args)

    logger.debug("Loading space")
    space = load_numpy(args.space).normalize()

    # Handle vocabulary issues
    logger.debug("Reading data")
    data = pd.read_table("data2/%s.tsv" % args.data)
    data['word1'] = data['word1'].apply(lambda x: best_pos_match(x, space))
    data['word2'] = data['word2'].apply(lambda x: best_pos_match(x, space))

    mask1 = data.word1.apply(lambda x: x in space.lookup)
    mask2 = data.word2.apply(lambda x: x in space.lookup)

    T = len(data)
    M1T = np.sum(mask1)
    M2T = np.sum(mask2)
    data = data[mask1 & mask2].reset_index(drop=True)
    F = len(data)
    logger.debug("")
    logger.debug("    Total Data: %5d" % T)
    logger.debug("       LHS OOV: %5d ( %4.1f%% )" % (M1T, M1T*100./T))
    logger.debug("       RHS OOV: %5d ( %4.1f%% )" % (M2T, M2T*100./T))
    logger.debug("         Final: %5d ( %4.1f%% )" % (F, F*100./T))
    logger.debug("")

    # dwp = data with predictions
    dwp = data.copy()

    logger.debug("         Model: %s" % args.model)
    model, features = models.load_setup(args.model)

    PROB = hasattr(model, 'predict_proba')

    logger.debug("      Features: %s" % features)
    rng = np.random.RandomState(31337)
    X, y = models.generate_feature_matrix(data, space, features)

    # need our folds for cross validation
    folds = fold.generate_folds_column(rng, data, 'category')
    train_sizes = np.array([len(f[0]) for f in folds], dtype=np.float)
    test_sizes = np.array([len(f[1]) for f in folds], dtype=np.float)

    scores = []
    for foldno, (train, test) in enumerate(folds):
        #logger.debug("   ... fold %2d/%2d" % (foldno, N_FOLDS))
        # generate features
        train_X, train_y = X[train], y[train]
        test_X, test_y = X[test], y[test]
        model.fit(train_X, train_y)
        preds_y = model.predict(test_X)

        scores.append([metrics.f1_score(test_y, preds_y),
                        metrics.precision_score(test_y, preds_y),
                        metrics.recall_score(test_y, preds_y),
                        metrics.accuracy_score(test_y, preds_y)])

    f1, p, r, a = np.mean(scores, axis=0)
    f1s, ps, rs, s = np.std(scores, axis=0)
    logger.info(" fin F1 pooled: %.3f   %.3f" % (f1, f1s))
    logger.info(" fin  P pooled: %.3f   %.3f" % (p, ps))
    logger.info(" fin  R pooled: %.3f   %.3f" % (r, rs))
    logger.info(" fin  A pooled: %.3f   %.3f" % (a, s))

if __name__ == '__main__':
    main()
