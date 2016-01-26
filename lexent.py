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

N_FOLDS = 20

# before anything else, configure the logger
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s   %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def main():
    parser = argparse.ArgumentParser('Lexical Entailment Classifier')
    parser.add_argument('--data', '-d', help='Input file')
    parser.add_argument('--space', '-s', help='Distributional space')
    parser.add_argument('--seed', '-S', default=1, type=int, help='Random seed')
    parser.add_argument('--model', '-m', help='Model setup', choices=models.SETUPS.keys())
    parser.add_argument('--experiment', '-e', default='standard', choices=('standard', 'match_error'))
    parser.add_argument('--output', '-o', default='results')
    args = parser.parse_args()

    logger.debug('Lexent Arguments: ')
    logger.debug(args)

    rng = np.random.RandomState(args.seed)

    logger.debug("Loading space")
    space = load_numpy(args.space).normalize()

    # Handle vocabulary issues
    logger.debug("Reading data")
    data = pd.read_table("data/%s/data.tsv" % args.data, header=None, names=('word1', 'word2', 'entails'))
    data['word1'] = data['word1'].apply(lambda x: x.lower() + '/NN')
    data['word2'] = data['word2'].apply(lambda x: x.lower() + '/NN')

    mask1 = data.word1.apply(lambda x: x in space.lookup)
    mask2 = data.word2.apply(lambda x: x in space.lookup)


    T = len(data)
    M1T = np.sum(mask1)
    M2T = np.sum(mask2)
    logger.debug("")
    logger.debug("Total Data: %6d" % T)
    logger.debug("   LHS OOV: %6d ( %4.1f%% )" % (M1T, M1T*100./T))
    logger.debug("   RHS OOV: %6d ( %4.1f%% )" % (M2T, M2T*100./T))
    data = data[mask1 & mask2].reset_index(drop=True)
    F = len(data)
    logger.debug("     Final: %6d ( %4.1f%% )" % (F, F*100./T))
    logger.debug("")

    logger.debug("Generating %d folds..." % N_FOLDS)

    # need our folds for cross validation
    folds = fold.generate_folds_lhs(rng, data, n_folds=N_FOLDS)
    train_sizes = np.array([len(f[0]) for f in folds], dtype=np.float)
    test_sizes = np.array([len(f[1]) for f in folds], dtype=np.float)

    logger.debug("Training sizes: %.1f" % np.mean(train_sizes))
    logger.debug("    Test sizes: %.1f" % np.mean(test_sizes))
    logger.debug(" Test-Tr ratio: %.1f%%" % np.mean(test_sizes*100./(train_sizes + test_sizes)))
    logger.debug(" Percent data: %.1f%%" % np.mean((train_sizes + test_sizes)*100./F))
    logger.debug("")

    logger.debug("Setting up the model:")
    model, features = models.load_setup(args.model)


    # dwp = data with predictions
    dwp = data.copy()
    dwp['prediction'] = False
    dwp['fold'] = -1

    logger.debug("Generating features")
    X, y = models.generate_feature_matrix(data, space, features)

    # perform cross validation
    logger.debug("Performing experiment: %s" % args.experiment)
    scores = []
    for foldno, (train, test) in enumerate(folds):
        logger.debug("   ... fold %2d/%2d" % (foldno, N_FOLDS))
        # generate features
        train_X, train_y = X[train], y[train]
        test_X, test_y = X[test], y[test]

        model.fit(train_X, train_y)
        preds_y = model.predict(test_X)
        dwp.loc[test,'prediction'] = preds_y
        dwp.loc[test,'fold'] = foldno
        scores.append(metrics.f1_score(test_y, preds_y))

    logger.info("F1 across CV: %.3f" % np.mean(scores))
    logger.info("         std: %.3f" % np.std(scores))
    logger.info("   F1 pooled: %.3f" % metrics.f1_score(dwp['entails'], dwp['prediction']))


    dwp.to_csv("%s/exp:%s,data:%s,space:%s,model:%s,seed:%d.csv" % (
        args.output, args.experiment, args.data,
        os.path.basename(args.space),
        args.model, args.seed
        ), index=False)

    if len(dwp[dwp['fold'] == -1]) != 0:
        logger.error("Some of the data wasn't predicted!\n" +
                     dwp[dwp['fold'] == -1])

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
