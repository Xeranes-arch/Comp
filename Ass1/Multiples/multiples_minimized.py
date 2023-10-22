import numpy as np


def main():
    a = 3
    b = 5

    arr = np.array([a * i for i in range(1, 400)])
    arr = np.append(arr, np.array([b * i for i in range(1, 201)]))
    arr = np.delete(arr, (arr >= 1000))

    print(arr)
    arr = np.unique(arr)
    print(arr)

    return arr.sum()


if __name__ == "__main__":
    print(main())
