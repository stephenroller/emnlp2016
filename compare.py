#!/usr/bin/env python
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy import stats

def siglevel(p):
    if p < .001:
        return '***'
    if p < .01:
        return '**'
    if p < .05:
        return '*'
    return ''

def mcnemar(A,B, C,D, alpha= 0.05, onetailed = False,verbose= False):
    """
    Performs a mcnemar test.
       A,B,C,D- counts in the form
        A    B  A+B
        C    D  C+D
       A+C  B+D  n

       alpha - level of significance
       onetailed -False for two-tailed test
                  True for one-tailed test
    Returns True if Null hypotheses pi1 == pi2 is accepted
    else False.
    """
    tot = float(A + B + C + D)
    Z = (B-C)/ np.sqrt(B+C)

    if verbose:
        print "McNemar Test with A,B,C,D = ", A,B, C,D
        print "Ratios:p1, p2 = ",(A+B)/tot, (C + D) /tot
        print "Z test statistic Z = ", Z


    if onetailed:
       if (B-C> 0):
         zcrit2 = stats.norm.ppf(1-alpha)
         result = True if (Z < zcrit2)else False
         if verbose:
            print "Upper critical value=", zcrit2
            print "Decision:",  "Accept " if (result) else "Reject ",
            print "Null hypothesis at alpha = ", alpha
       else:
         zcrit1 = stats.norm.ppf(alpha)
         result = False if (Z < zcrit1) else False
         if verbose:
            print "Lower critical value=", zcrit1
            print "Decision:",  "Accept " if (result) else "Reject ",
            print "Null hypothesis at alpha = ", alpha


    else:
       zcrit1 = stats.norm.ppf(alpha/2.0)
       zcrit2 = stats.norm.ppf(1-alpha/2.0)

       result = True if (zcrit1 < Z < zcrit2) else False
       if verbose:
          print "Lower and upper critical limits:", zcrit1, zcrit2
          print "Decision:","Accept " if result else "Reject ",
          print "Null hypothesis at alpha = ", alpha

    return result



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
    assert np.all(gold == np.array(t2.entails))
    assert np.all(t1.foldno == t2.foldno)

    joiner = None
    if 'bless' in args.input1:
        joiner = 'orig/joined/bless2011.tsv'
    if 'turney' in args.input1:
        joiner = 'orig/joined/turney2014.tsv'
    if joiner:
        tojoin = pd.read_table(joiner)
        catlookup = {(w1 + '/NN', w2 + '/NN'): c for w1, w2, c in zip(tojoin.word1, tojoin.word2, tojoin.category)}
        t1['category'] = [catlookup.get((w1, w2), None) for w1, w2 in zip(t1.word1, t1.word2)]
        if 'rel' in tojoin.columns:
            rellookup = {(w1 + '/NN', w2 + '/NN'): r for w1, w2, r in zip(tojoin.word1, tojoin.word2, tojoin.rel)}
            t1['rel'] = [rellookup.get((w1, w2), None) for w1, w2 in zip(t1.word1, t1.word2)]


    scores = {k: [] for k in ['f1A', 'f1B', 'pA', 'pB', 'rA', 'rB', 'same']}

    pred1 = np.array(t1['prediction'])
    pred2 = np.array(t2['prediction'])
    for fold in t1.foldno.unique():
        m = (t1.foldno == fold).values
        goldm = gold[m]
        pred1m = pred1[m]
        pred2m = pred2[m]
        scores['f1A'].append(f1_score(goldm, pred1m))
        scores['f1B'].append(f1_score(goldm, pred2m))
        scores['pA'].append(precision_score(goldm, pred1m))
        scores['pB'].append(precision_score(goldm, pred2m))
        scores['rA'].append(recall_score(goldm, pred1m))
        scores['rB'].append(recall_score(goldm, pred2m))
        scores['same'].append(np.mean(pred1m == pred2m))

    correct1 = np.array(t1['correct'])
    correct2 = np.array(t2['correct'])
    #t, p = stats.ttest_rel(scores['f1A'], scores['f1B'])
    both_right   = np.sum( correct1 &  correct2)
    both_wrong   = np.sum(~correct1 & ~correct2)
    first_right  = np.sum( correct1 & ~correct2)
    second_right = np.sum(~correct1 &  correct2)

    mcsig = mcnemar(both_right, first_right, second_right, both_wrong, .05, False, False)

    print "SystemA: %s" % args.input1
    print "SystemB: %s" % args.input2
    print "     F1: %.3f vs %.3f" % (np.mean(scores['f1A']), np.mean(scores['f1B']))
    print "      P: %.3f vs %.3f" % (np.mean(scores['pA']), np.mean(scores['pB']))
    print "      R: %.3f vs %.3f" % (np.mean(scores['rA']), np.mean(scores['rB']))
    print "   Same: %.3f" % np.mean(scores['same'])
    print "mcnemar: %s" % mcsig
    #print "t(A!=B): %.3f %s" % (p, siglevel(p))
    #print "  boots: %.3f %s" % (pbs, siglevel(pbs))

    print "Comparing examples:"
    mask = correct1 & ~correct2
    print "First right, second wrong [%d]:" % np.sum(mask)
    print "%d were entailing, %d nonentailing" % (np.sum(mask & t1.entails.values), np.sum(mask & ~t1.entails.values))
    if joiner:
        if 'rel' in t1.columns:
            for r in t1.rel.unique():
                maskc = t1.rel == r
                print "  rel %-20s: %d ; %d" % (r, np.sum(mask & maskc & t1.entails.values), np.sum(mask & maskc & ~t1.entails.values))
            print
        for c in t1.category.unique():
            maskc = t1.category == c
            print "  cat %-20s: %d ; %d" % (c, np.sum(mask & maskc & t1.entails.values), np.sum(mask & maskc & ~t1.entails.values))
    #t1[mask].to_csv(sys.stdout, sep="\t", index=False, columns=('word1', 'word2', 'entails'))
    print
    mask = ~correct1 & correct2
    print "First wrong, second right [%d]:" % second_right
    print "%d were entailing, %d nonentailing" % (np.sum(mask & t1.entails.values), np.sum(mask & ~t1.entails.values))
    if joiner:
        if 'rel' in t1.columns:
            for r in t1.rel.unique():
                maskc = t1.rel == r
                print "  rel %-20s: %d ; %d" % (r, np.sum(mask & maskc & t1.entails.values), np.sum(mask & maskc & ~t1.entails.values))
            print
        for c in t1.category.unique():
            maskc = t1.category == c
            print "  cat %-20s: %d ; %d" % (c, np.sum(mask & maskc & t1.entails.values), np.sum(mask & maskc & ~t1.entails.values))
    #t1[mask].to_csv(sys.stdout, sep="\t", index=False, columns=('word1', 'word2', 'entails'))



if __name__ == '__main__':
    main()
