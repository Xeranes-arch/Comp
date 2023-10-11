import numpy as np


def prompt(name):
    """Gets value from user to work with."""
    flag = False
    while flag == False:
        try:
            n = input("Pick number " + name + ":")
            flag = int(n) in np.arange(1, 11)
            if flag == True:
                break
            print("Input natural numbers between one and ten.")

        except:
            print("That's just not a number bro.")

    return int(n)


def main():
    a = prompt("one")
    if a == 1:
        arr1 = np.array(1)
    else:
        arr1 = np.array([a**i for i in range(1, 11)])

    b = prompt("two")
    if b == 1:
        arr2 = np.array(1)
    else:
        arr2 = np.array([b**i for i in range(1, 11)])

    arr = np.append(arr1, arr2)
    arr = np.delete(arr, (arr > 1000))
    print(arr.sum())


if __name__ == main():
    main()
