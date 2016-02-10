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

def standard_experiment(data, X, Y):
    pass

def cv_trials(X, y, folds, model):
    pass

def main():
    # Argument parsing
    parser = argparse.ArgumentParser('Lexical Entailment Classifier')
    parser.add_argument('--data', '-d', help='Input file')
    parser.add_argument('--space', '-s', help='Distributional space')
    parser.add_argument('--model', '-m', help='Model setup', choices=models.SETUPS.keys())
    parser.add_argument('--experiment', '-e', default='standard', choices=('standard', 'artificial', 'match_error', 'featext', 'strat'))
    parser.add_argument('--stratifier')
    parser.add_argument('--output', '-o')
    parser.add_argument('--tuning', '-t', action='store_true')
    args = parser.parse_args()

    logger.debug('Lexent Arguments: ')
    logger.debug(args)

    # Steps that are the same regardless of experiments
    logger.debug("Loading space")
    nonnorm = load_numpy(args.space)
    space = nonnorm.normalize()

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

    PROB = hasattr(model, 'predict_proba')
    # dwp = data with predictions
    dwp = data.copy()

    pooled_evals = []
    if PROB:
        pooled_evals_prob = []

    for seed in xrange(1, 21):
        logger.debug("       On seed: %d / %d" % (seed, 20))
        logger.debug("  Genenerating: %d folds" % N_FOLDS)
        rng = np.random.RandomState(seed)
        fold_key = 'fold_%02d' % seed
        pred_key = 'prediction_%02d' % seed
        pred_prob_key = 'probability_%02d' % seed
        dwp[pred_key] = False
        dwp[fold_key] = -1

        # need our folds for cross validation
        folds = fold.generate_folds_lhs(rng, data, n_folds=N_FOLDS)
        train_sizes = np.array([len(f[0]) for f in folds], dtype=np.float)
        test_sizes = np.array([len(f[1]) for f in folds], dtype=np.float)

        logger.debug("Training sizes: %.1f" % np.mean(train_sizes))
        logger.debug("    Test sizes: %.1f" % np.mean(test_sizes))
        logger.debug(" Test-Tr ratio: %.1f%%" % np.mean(test_sizes*100./(train_sizes + test_sizes)))
        logger.debug("  Percent data: %.1f%%" % np.mean((train_sizes + test_sizes)*100./F))

        # perform cross validation
        logger.debug("    Experiment: %s" % args.experiment)
        scores = []
        for foldno, (train, test) in enumerate(folds):
            #logger.debug("   ... fold %2d/%2d" % (foldno, N_FOLDS))
            # generate features
            train_X, train_y = X[train], y[train]
            test_X, test_y = X[test], y[test]
            model.fit(train_X, train_y)
            preds_y = model.predict(test_X)

            if PROB:
                train_prob = model.predict_proba(train_X)[:,1]
                proba_y = model.predict_proba(test_X)[:,1]
                dwp.loc[test,pred_prob_key] = proba_y
                if args.tuning:
                    tc.fit(np.array([train_prob]).T, train_y)
                    #logger.debug("     Threshold: %.3f%%" % tc.threshold_)
                    adjusted = tc.predict(np.array([proba_y]).T)
                    preds_y = adjusted

            dwp.loc[test,pred_key] = preds_y
            dwp.loc[test,fold_key] = foldno
            scores.append([metrics.f1_score(test_y, preds_y),
                           metrics.precision_score(test_y, preds_y),
                           metrics.recall_score(test_y, preds_y),
                           metrics.accuracy_score(test_y, preds_y)])

        if len(dwp[dwp[fold_key] == -1]) != 0:
            logger.error("Some of the data wasn't predicted!\n" + dwp[dwp[fold_key] == -1])


        pooled_f1 = metrics.f1_score(dwp['entails'], dwp[pred_key])
        pooled_p = metrics.precision_score(dwp['entails'], dwp[pred_key])
        pooled_r = metrics.recall_score(dwp['entails'], dwp[pred_key])
        pooled_a = metrics.accuracy_score(dwp['entails'], dwp[pred_key])

        pooled_evals.append([pooled_f1, pooled_p, pooled_r, pooled_a])

        if PROB:
            pooled_evals_prob.append(
                [metrics.average_precision_score(dwp['entails'], dwp[pred_prob_key]),
                 metrics.roc_auc_score(dwp['entails'], dwp[pred_prob_key])])

        f1, p, r, a = np.mean(scores, axis=0)
        f1s, ps, rs, s = np.std(scores, axis=0)

        logger.info("  F1 across CV: %.3f   %.3f" % (f1, f1s))
        logger.info("   P across CV: %.3f   %.3f" % (p, ps))
        logger.info("   R across CV: %.3f   %.3f" % (r, rs))
        logger.info("   A across CV: %.3f   %.3f" % (a, s))
        logger.debug("")

    f1, p, r, a = np.mean(pooled_evals, axis=0)
    f1s, ps, rs, s = np.std(pooled_evals, axis=0)
    logger.info(" fin F1 pooled: %.3f   %.3f" % (f1, f1s))
    logger.info(" fin  P pooled: %.3f   %.3f" % (p, ps))
    logger.info(" fin  R pooled: %.3f   %.3f" % (r, rs))
    logger.info(" fin  A pooled: %.3f   %.3f" % (a, s))
    if PROB:
        ap, roc = np.mean(pooled_evals_prob, axis=0)
        aps, rocs = np.std(pooled_evals_prob, axis=0)
        logger.info(" fin AP pooled: %.3f   %.3f" % (ap, aps))
        logger.info(" finROC pooled: %.3f   %.3f" % (roc, rocs))

    logger.debug("")
    logger.debug("Analyzing top weighted features:")

    logger.debug("")

    if args.output:
        dwp.to_csv("%s/exp:%s,data:%s,space:%s,model:%s.csv.bz2" % (
            args.output, args.experiment, args.data,
            os.path.basename(args.space),
            args.model
            ), index=False, compression='bz2')

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




if __name__ == '__main__':
    main()
