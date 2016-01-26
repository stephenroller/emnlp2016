#!/usr/bin/env python
import sys
import argparse
import numpy as np
import pandas as pd

ignore = set([
    'training set sizes',
    'testing set sizes',
    'test as % of train',
    'Standard Classification',
    'always',
    'lhs',
    'rhs',
])

def parse_item(lines):
    # contains online lines between SPACE: and the ending trial
    space = ''
    data = ''
    for line in lines:
        if 'SPACE:' in line:
            space = line.split()[-1]
        elif 'DATA:' in line:
            data = line.split()[-1]
        else:
            model, feats, f1, lower, leq1, mean, leq2, upper = line.split()
            yield dict(space=space, data=data, model=model + "_" + feats,
                       lower=float(lower), upper=float(upper), mean=float(mean))



def parse_items(lines):
    grouped = []
    for line in lines:
        if 'SPACE:' in line:
            if grouped:
                for i in parse_item(grouped): yield i
                grouped = []
        grouped.append(line)
    if grouped:
        for i in parse_item(grouped): yield i


def main():
    parser = argparse.ArgumentParser('Collects results neatly')
    parser.add_argument('--input', '-i', default='results.log')
    parser.add_argument('--space', '-s')
    parser.add_argument('--model', '-m')
    parser.add_argument('--data', '-d')
    args = parser.parse_args()


    # infer the groupby variable
    s, d, m = bool(args.space), bool(args.data), bool(args.model)
    assert not (s and d and m)
    assert (s or d or m)
    assert not (s ^ d ^ m)
    groupby = ['space', 'data', 'model'][np.argmin([s, d, m])]


    # first we want to just filter noise
    skipping = False
    keepers = []
    with open(args.input) as results:
        for line in results:
            line = line.strip()
            if not line: continue
            if any(i in line for i in ignore): continue

            if line == 'False Positive Issue:':
                skipping = True
            elif line.startswith('SPACE:'):
                skipping = False

            if not skipping:
                keepers.append(line)

    # now parse out the actual items
    trials = parse_items(keepers)

    # and filter them according to the arguments
    present = []
    for t in trials:
        if args.space and args.space not in t['space']: continue
        if args.data and args.data not in t['data']: continue
        if args.model and args.model not in t['model']: continue
        present.append(t)

    if not present:
        raise ValueError("Got no items, fool!")

    df = pd.DataFrame(present)
    df = df.groupby(groupby).aggregate({'mean': [np.min, np.mean, np.max]})
    print df


if __name__ == '__main__':
    main()
