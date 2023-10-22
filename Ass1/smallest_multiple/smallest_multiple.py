import numpy as np


def main():
    a = np.arange(1, 21)
    a[(a % 2).astype(bool)] = a[(a % 2).astype(bool)] * 2
    print(a)
    prod = np.prod(a)
    return prod


if __name__ == "__main__":
    print(main())
