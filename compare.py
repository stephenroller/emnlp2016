#!/usr/bin/env python
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

def main():
    parser = argparse.ArgumentParser('Compare the predictions of two systems')
    parser.add_argument('input1')
    parser.add_argument('input2')
    args = parser.parse_args()

    t1 = pd.read_csv(args.input1)
    t2 = pd.read_csv(args.input2)
    assert len(t1) == len(t2)

    assert np.all(t1.entails == t2.entails)
    gold = np.array(t1.entails)

    columns = [c for c in t1.columns if c.startswith('prediction')]

    scores = {'f1A': [], 'pA': [], 'rA': [], 'same': [],
              'f1B': [], 'pB': [], 'rB': []}

    biggold = []
    predA = []
    predB = []

    for col in columns:
        pred1 = np.array(t1[col])
        pred2 = np.array(t2[col])
        scores['f1A'].append(f1_score(gold, pred1))
        scores['f1B'].append(f1_score(gold, pred2))
        scores['pA'].append(precision_score(gold, pred1))
        scores['pB'].append(precision_score(gold, pred2))
        scores['rA'].append(recall_score(gold, pred1))
        scores['rB'].append(recall_score(gold, pred2))
        scores['same'].append(np.mean(pred1 == pred2))

        predA += list(pred1)
        predB += list(pred2)
        biggold += list(gold)

    print "SystemA: %s" % args.input1
    print "SystemB: %s" % args.input2
    print "     F1: %.3f vs %.3f" % (np.mean(scores['f1A']), np.mean(scores['f1B']))
    print "      P: %.3f vs %.3f" % (np.mean(scores['pA']), np.mean(scores['pB']))
    print "      R: %.3f vs %.3f" % (np.mean(scores['rA']), np.mean(scores['rB']))
    print "   Same: %.3f" % np.mean(scores['same'])

        #records['entails'] += records['entails']
        #records['m1pred'] += t1[col]
        #records['m2pred'] += t2[col]

        #m = (pred1 == pred2) | (pred1 != pred2)

        #records['word1'] += list(t1[m]['word1'])
        #records['word2'] += list(t1[m]['word2'])
        #records['entails'] += list(t1[m]['entails'])
        #records['m1pred'] += list(t1[m][col])
        #records['m2pred'] += list(t2[m][col])



if __name__ == '__main__':
    main()
