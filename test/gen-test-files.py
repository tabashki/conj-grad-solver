#!/usr/bin/python3

import sys
import numpy as np


def sparse(n):
    return np.diag([-2]*n) + np.diag([1]*(n-1),1) + np.diag([1]*(n-1),-1)


def testfiles(n,d):
    a = sparse(n)
    x = np.random.random_sample(n) - 0.5
    x = x * 10
    b = np.matmul(a, x)
    with open(f'{d}.txt', 'w') as f:
        for bb in b:
            print(f'{bb:.10e}', file=f)
    with open(f'{d}.expected.txt', 'w') as f:
        for xx in x:
            print(f'{xx:.10e}', file=f)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Insufficient arguments')
        print('gen_test_files.py [N] [FILENAME]')
        sys.exit(1)
    n = int(sys.argv[1])
    d = sys.argv[2]
    testfiles(n, d)
    sys.exit(0)
