#!/usr/bin/env python

import numpy as np
from sklearn.preprocessing import normalize

class VectorSpace(object):
    def __init__(self, matrix, vocab):
        self.vocab = vocab
        self.matrix = matrix
        self.lookup = {v:i for i, v in enumerate(vocab)}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.matrix[key]
        elif isinstance(key, str):
            return self.matrix[self.lookup[key]]

    def __contains__(self, key):
        return key in self.lookup

    def save_mikolov_text(self, filename):
        with open(filename, 'w') as f:
            for i, word in enumerate(self.vocab):
                row = self.matrix[i]
                line = " ".join(map(str, row))
                f.write("%s %s\n" % (word, line))

    def subset(self, whitelist):
        whitelist = set(whitelist)
        keep = [(i, word) for i, word in enumerate(self.vocab) if word in whitelist]
        indices, newvocab = zip(*keep)
        newmatrix = self.matrix[indices,:]
        return VectorSpace(newmatrix, newvocab)

    def normalize(self):
        return VectorSpace(normalize(self.matrix, norm='l2', axis=1), self.vocab)



def load_mikolov_text(filename):
    vocab = []
    matrix = []
    with open(filename) as f:
        for l in f:
            l = l.strip().split()
            vocab.append(l.pop(0))
            matrix.append(np.array(map(float, l)))

    matrix = np.array(matrix)
    return VectorSpace(matrix, vocab)


if __name__ == '__main__':
    space = load_mikolov_text(SPACE)
    car = space['car']
    truck = space['truck']
    c = np.dot(car, car)
    t = np.dot(truck, truck)
    print np.dot(car, truck)/np.sqrt(c * t)

