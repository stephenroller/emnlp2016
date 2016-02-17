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

def consolidate(list_of_dicts):
    """
    Returns a dict of lists from a list of dicts
    """
    consolidated = {}
    for k in list_of_dicts[0].keys():
        consolidated[k] = np.array([d[k] for d in list_of_dicts])
    return consolidated

def feature_extraction(X, y, model, space):
    np.set_printoptions(suppress=True)
    n_output = 30
    model.fit(X, y)
    D = space.matrix.shape[1]
    segments = int(X.shape[1] / D)
    full_magn = np.sum(np.square(model.coef_))
    for s in xrange(segments):
        l = D*s
        r = D*(s+1)
        feats = model.coef_[0,l:r]
        p = np.sum(np.square(feats)) / full_magn
        print "Segment #%d [%d-%d] [%2.1f%%]" % (s + 1, l, r, 100*p)
        print feats
        word_ranks = space.matrix.dot(feats)
        sorted = word_ranks.argsort()
        print "  Strongest words:"
        for i in xrange(n_output):
            idx = sorted[-(i+1)]
            score = word_ranks[idx]
            word = space.vocab[idx]
            print "    %6.3f  %s" % (score, word)
        print "  Weakest words:"
        for i in xrange(n_output-1, -1, -1):
            idx = sorted[i]
            score = word_ranks[idx]
            word = space.vocab[idx]
            print "    %6.3f  %s" % (score, word)
        ctx_ranks = space.cmatrix.dot(feats)
        sorted = ctx_ranks.argsort()
        print "  Strongest contexts:"
        for i in xrange(n_output):
            idx = sorted[-(i+1)]
            score = ctx_ranks[idx]
            word = space.cvocab[idx]
            print "    %6.3f  %s" % (score, word)
        print "  Weakest contexts:"
        for i in xrange(n_output-1, -1, -1):
            idx = sorted[i]
            score = ctx_ranks[idx]
            word = space.cvocab[idx]
            print "    %6.3f  %s" % (score, word)




def standard_experiment(data, X, y, model, tuning=False):
    with_probas = hasattr(model, 'predict_proba')

    # data with predictions
    dwp = data.copy()

    pooled_evals = []

    for seed in xrange(1, 21):
        logger.debug("       On seed: %d / %d" % (seed, 20))
        logger.debug("  Genenerating: %d folds" % N_FOLDS)
        rng = np.random.RandomState(seed)
        fold_key = 'fold_%02d' % seed
        pred_key = 'prediction_%02d' % seed
        pred_prob_key = 'probability_%02d' % seed

        # need our folds for cross validation
        folds = fold.generate_folds_lhs(rng, data, n_folds=N_FOLDS)
        train_sizes = np.array([len(f[0]) for f in folds], dtype=np.float)
        test_sizes = np.array([len(f[1]) for f in folds], dtype=np.float)

        logger.debug("Training sizes: %.1f" % np.mean(train_sizes))
        logger.debug("    Test sizes: %.1f" % np.mean(test_sizes))
        logger.debug(" Test-Tr ratio: %.1f%%" % np.mean(test_sizes*100./(train_sizes + test_sizes)))
        logger.debug("  Percent data: %.1f%%" % np.mean((train_sizes + test_sizes)*100./len(y)))

        # perform cross validation

        pooled_eval, predictions, cv_scores = cv_trials(X, y, folds, model, tuning=tuning)
        pooled_evals.append(pooled_eval)

        for k in cv_scores.keys():
            mu = cv_scores[k].mean()
            sigma = cv_scores[k].std()
            logger.info(" %3s across CV: %.3f   %.3f" % (k.upper(), mu, sigma))
        logger.debug("")
        #if len(dwp[dwp[fold_key] == -1]) != 0:
        #    logger.error("Some of the data wasn't predicted!\n" + dwp[dwp[fold_key] == -1])


    pooled_evals = consolidate(pooled_evals)
    for k in pooled_evals.keys():
        mu = pooled_evals[k].mean()
        sigma = pooled_evals[k].std()
        logger.info(" fin%3s pooled: %.3f   %.3f" % (k.upper(), mu, sigma))

    logger.debug("")

    if False and args.output:
        dwp.to_csv("%s/exp:%s,data:%s,space:%s,model:%s.csv.bz2" % (
            args.output, args.experiment, args.data,
            os.path.basename(args.space),
            args.model
            ), index=False, compression='bz2')

def cv_trials(X, y, folds, model, tuning=False):
    with_probas = hasattr(model, 'predict_proba')
    tuning = with_probas and tuning # force tuning off if there aren't probas
    N = len(y)

    cv_scores = []
    predictions = {
        'pred': np.zeros(N, dtype=np.bool),
        'adj_pred': np.zeros(N, dtype=np.bool),
        'proba': np.zeros(N),
        'foldno': np.zeros(N, dtype=np.int32) - 1,
    }

    for foldno, (train, test) in enumerate(folds):
        train_X, train_y = X[train], y[train]
        test_X, test_y = X[test], y[test]
        model.fit(train_X, train_y)
        preds_y = model.predict(test_X)
        predictions['pred'][test] = preds_y

        if with_probas:
            train_prob = model.predict_proba(train_X)[:,1]
            proba_y = model.predict_proba(test_X)[:,1]
            predictions['proba'][test] = proba_y
            if tuning:
                tc.fit(np.array([train_prob]).T, train_y)
                adjusted = tc.predict(np.array([proba_y]).T)
                predictions['adj_pred'][test] = adjusted

        predictions['foldno'][test] = foldno

        fold_eval = {'f1': metrics.f1_score(test_y, preds_y),
                      'p': metrics.precision_score(test_y, preds_y),
                      'r': metrics.recall_score(test_y, preds_y),
                      'a': metrics.accuracy_score(test_y, preds_y)}

        if with_probas:
            fold_eval['ap'] = metrics.average_precision_score(test_y, proba_y)
            fold_eval['roc'] = metrics.roc_auc_score(test_y, proba_y)

        cv_scores.append(fold_eval)

    # now we want to compute global evaluations, and consolidate metrics
    cv_scores = consolidate(cv_scores)

    if tuning:
        preds_y = predictions['adj_pred']
    else:
        preds_y = predictions['pred']
    pooled_eval = {'f1': metrics.f1_score(y, preds_y),
                    'p': metrics.precision_score(y, preds_y),
                    'r': metrics.recall_score(y, preds_y),
                    'a': metrics.accuracy_score(y, preds_y)}
    if with_probas:
        proba_y = predictions['proba']
        pooled_eval['ap'] = metrics.average_precision_score(y, proba_y)
        pooled_eval['roc'] = metrics.roc_auc_score(y, proba_y)

    return pooled_eval, predictions, cv_scores


def main():
    # Argument parsing
    parser = argparse.ArgumentParser('Lexical Entailment Classifier')
    parser.add_argument('--data', '-d', help='Input file')
    parser.add_argument('--space', '-s', help='Distributional space')
    parser.add_argument('--model', '-m', help='Model setup', choices=models.SETUPS.keys())
    parser.add_argument('--experiment', '-e', default='standard', choices=('standard', 'artificial', 'match_error', 'featext', 'strat'))
    parser.add_argument('--stratifier')
    #parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--output', '-o')
    parser.add_argument('--tuning', '-t', action='store_true')
    args = parser.parse_args()

    logger.debug('Lexent Arguments: ')
    logger.debug(args)

    # Steps that are the same regardless of experiments
    logger.debug("Loading space")
    space = load_numpy(args.space).normalize()

    # Handle vocabulary issues
    logger.debug("Reading data")
    data = pd.read_table("data/%s/data.tsv" % args.data, header=None, names=('word1', 'word2', 'entails'))
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

    #if args.experiment == 'artificial':
    #    np.random.seed(31337)
    #    exists = {}
    #    for i, row in data.iterrows():
    #        exists[(row['word1'], row['word2'])] = row['entails']
    #    fake_data = data.copy()
    #    word2s = np.array(fake_data['word2'])
    #    np.random.shuffle(word2s)
    #    fake_data['word2'] = word2s
    #    fake_data['entails'] = False
    #    fake_mask = []
    #    for i, row in fake_data.iterrows():
    #        fake_mask.append((row['word1'], row['word2']) not in exists)
    #    data = pd.concat([data, fake_data[fake_mask]], ignore_index=True)

    logger.debug("         Model: %s" % args.model)
    model, features = models.load_setup(args.model)

    logger.debug("      Features: %s" % features)
    X, y = models.generate_feature_matrix(data, space, features)

    if args.experiment == 'standard':
        standard_experiment(data, X, y, model, tuning=args.tuning)
    elif args.experiment == 'featext':
        feature_extraction(X, y, model, space)





if __name__ == '__main__':
    main()
