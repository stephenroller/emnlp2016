#!/usr/bin/env python
import sys
import argparse
import numpy as np
from scipy.linalg import svd

from space import load_mikolov_text, VectorSpace

def main():
    parser = argparse.ArgumentParser('Transforms the space by applying the SVD')
    parser.add_argument('--input', '-i', help='Input file')
    parser.add_argument('--output', '-o', help='Output file')
    args = parser.parse_args()

    space = load_mikolov_text(args.input)
    U, s, Vh = svd(space.matrix, full_matrices=False)
    transformed = U.dot(np.diag(s))

    newspace = VectorSpace(transformed, space.vocab)
    newspace.save_mikolov_text(args.output)


if __name__ == '__main__':
    main()
