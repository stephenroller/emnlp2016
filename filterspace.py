#!/usr/bin/env python
import sys
import argparse
from space import load_mikolov_text

def main():
    parser = argparse.ArgumentParser('Reduces a vector space to only the limited vocab.')
    parser.add_argument('--input', '-i', help='Input space')
    parser.add_argument('--output', '-o', help='Output space')
    parser.add_argument('--whitelist', '-w', help='Whitelist file.')
    parser.add_argument('--dims', '-d', type=int, help='Number of dimensions')
    args = parser.parse_args()


    space = load_mikolov_text(args.input)

    if args.whitelist:
        whitelist = []
        with open(args.whitelist) as wlf:
            for line in wlf:
                whitelist.append(line.strip().lower())
        whitelist = set(whitelist)
        filtered = space.subset(whitelist)
    else:
        filtered = space
    if args.dims:
        filtered.matrix = filtered.matrix[:,:args.dims]
    filtered.save_mikolov_text(args.output)

if __name__ == '__main__':
    main()

