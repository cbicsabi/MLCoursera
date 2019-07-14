import sys

import numpy as np

def cntr(x):
    if x == 0:
        return 0

    return factorial(x) + cntr(x)


def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n + factorial(n - 1)


def main():
    # x = np.random.rand(3, 3)
    # print(x)
    # for i in range(3):
    #     for j in range(3):
    #         print(x[i, j])

    x = cntr(10)
    print(x)
    x = factorial(10)
    print(x)


if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    main()
