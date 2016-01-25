"""
file   mcnemar.py
author Dr. Ernesto P. Adorio
       UPEPP at Clark Field, Pampanga
desc   Performs a nonparametric McNemar test for differences in two proportions.
"""


import sys
import pandas as pd
import numpy as np
from math  import sqrt
from scipy import stats

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
    Z = (B-C)/ sqrt(B+C)

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

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: sigtest.py [output1file] [output2file]'
    table1 = pd.read_csv(sys.argv[1])
    table2 = pd.read_csv(sys.argv[2])

    assert len(table1) == len(table2)
    table1['correct'] = table1['prediction'] == table1['entails']
    table2['correct'] = table2['prediction'] == table2['entails']

    together = pd.merge(table1, table2, on=('word1', 'word2'))

    both_right = np.sum((together.correct_x == True) & (together.correct_y == True))
    both_wrong = np.sum((together.correct_x == False) & (together.correct_y == False))
    second_right = np.sum((together.correct_x == False) & (together.correct_y == True))
    first_right = np.sum((together.correct_x == True) & (together.correct_y == False))

    mcnemar(both_right, first_right, second_right, both_wrong, 0.05, False, True)

