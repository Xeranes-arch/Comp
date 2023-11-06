import numpy as np
import matplotlib.pyplot as plt

N = 1
state = 1.0 * N, 0.0 * N, 0.0 * N

k = [10]
t = 0.001

lst = []


def update_exp_euler(st, k2, k1=1):
    d1 = -k1 * st[0]
    d2 = k1 * st[0] - k2 * st[1]
    d3 = k2 * st[1]
    return d1 * t, d2 * t, d3 * t


def exp_euler(state):
    lst = []
    for i in k:
        arr_exp_euler = np.array(state)
        old = arr_exp_euler

        j = 0
        while j <= 10000:
            j += 1
            new = np.add(old, update_exp_euler(old, i))
            arr_exp_euler = np.vstack((arr_exp_euler, new))
            old = new
        lst.append(arr_exp_euler)
    return lst


def heun(state):
    lst = []
    for i in k:
        arr_heun = np.array(state)
        old = arr_heun

        j = 0
        while j <= 10000:
            j += 1
            new_y = np.add(old, update_exp_euler(old, i))
            new_y2 = np.add(new_y, update_exp_euler(new_y, i))
            new = np.add(new_y, new_y2) * 1 / 2

            arr_heun = np.vstack((arr_heun, new))
            old = new

        lst.append(arr_heun)

    return lst


def graph(lst, a, c, l):
    for j, i in enumerate(lst):
        idx = np.arange(i[:, 0].size)

        plt.plot(idx, i[:, a], c=c, label=l[j])


lst1 = exp_euler(state)


lst2 = heun(state)
for a in [0, 1, 2]:
    graph(lst1, a, c="r", l=[None, None, None, "Euler"])
    graph(lst2, a, c="b", l=[None, None, None, "Heun"])

    plt.legend()
    plt.show()
