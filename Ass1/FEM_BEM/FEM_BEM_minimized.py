import numpy as np


def main():
    a = np.arange(1, 101)

    fem = 3
    bem = 5

    mask1 = np.invert((a % fem).astype(bool))
    mask2 = np.invert((a % bem).astype(bool))
    mask3 = mask1 * mask2

    a = a.astype(str)

    a[mask1] = "fem"
    a[mask2] = "bem"
    a[mask3] = "fem-bem"

    return a


if __name__ == "__main__":
    print(main())
