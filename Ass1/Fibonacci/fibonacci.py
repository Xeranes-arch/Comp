import numpy as np


def main():
    i = 2
    p = 1
    sum = 2
    limit = 4e6 * 0.618
    while i < limit:
        n = i + p
        p = i
        if not (n % 2):
            sum += n
        i = n
    return sum


if __name__ == "__main__":
    print(main())
