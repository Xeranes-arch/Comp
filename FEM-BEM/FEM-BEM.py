import numpy as np


def prompt(name, nrs):
    """Gets value from user to work with."""
    flag = False
    while flag == False:
        try:
            n = input("Pick a number as " + name + ":")
            flag = int(n) in nrs
        except:
            print("Yes, it has error handling.")
    return int(n)


def main():
    a = np.arange(1, 101)

    fem = prompt("fem", a)
    bem = prompt("bem", a)

    mask1 = np.invert((a % fem).astype(bool))
    mask2 = np.invert((a % bem).astype(bool))
    mask3 = mask1 * mask2

    a = a.astype(str)

    a[mask1] = "fem"
    a[mask2] = "bem"
    a[mask3] = "fem-bem"

    print(a)


if __name__ == main():
    main()
