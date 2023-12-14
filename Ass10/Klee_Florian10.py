import timeit
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from functions import *

np.random.seed()


def given_system():
    """Algorithms on given System visualized."""
    A = np.array([[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]]).astype(
        float
    )
    b = np.matrix([[1], [2], [3], [4]]).astype(float)
    x_0 = np.array([[0.1], [0.1], [0.1], [0.1]])
    sol = np.array(np.linalg.solve(A, b))

    res = steepest_decent_vis(A, b, x_0)
    print(type(res), type(sol))
    print("------------------------------------------\nSD SOLUTION:\n", res)
    err = max(abs((res - sol) / max(sol))) * 100
    print(
        "ERROR of: ",
        round(float(err[0]), 4),
        "%\n------------------------------------------",
    )

    res = CG_vis(A, b, x_0)
    print("------------------------------------------\nCG SOLUTION:\n", res)
    err = max(abs((res - sol) / max(sol))) * 100
    print(
        "ERROR of: ",
        round(float(err[0]), 4),
        "%\n------------------------------------------",
    )
    plt.legend(["SD", "CG"])
    plt.show()


def random_system(N):
    """Algorithms on random System visualized"""
    A, b, x_0, sol = genSys(N)

    res = steepest_decent_vis(A, b, x_0, cap=10000)
    print(type(res), type(sol))
    print("------------------------------------------\nSD SOLUTION:\n", res)
    err = max(abs((res - sol) / max(sol))) * 100
    print(
        "ERROR of: ",
        round(float(err[0]), 4),
        "%\n------------------------------------------",
    )

    res = CG_vis(A, b, x_0)
    print("------------------------------------------\nCG SOLUTION:\n", res)
    err = max(abs((res - sol) / max(sol))) * 100
    print(
        "ERROR of: ",
        round(float(err[0]), 4),
        "%\n------------------------------------------",
    )
    plt.legend(["SD", "CG"])
    plt.show()


def error_analysis_dec():
    lst = []
    N = np.arange(3, 10, 1)
    for i in N:
        print(i)
        y = 0
        n = 0
        for j in range(100):
            A, b, x_0, sol = genSys(i)
            try:
                res = steepest_decent(A, b, x_0, ctrl=True)
                y += 1
            except:
                n += 1
        lst.append(y / (y + n))
    print(N, lst)
    plt.plot(N, lst)
    plt.title("Rate of valid matrices that converge within 10000 Iterations")
    plt.show()


def error_analysis_cg():
    lst = []
    N = np.append(np.append(np.arange(3, 30), np.arange(30, 101, 15)), [200, 500, 1000])
    for i in N:
        print(i)
        y = 0
        n = 0
        err = 0
        for j in range(1000):
            A, b, x_0, sol = genSys(i)
            res = CG(A, b, x_0)
            err += max(abs((res - sol) / max(sol))) * 100
        lst.append(err / 1000)
    print(N, lst)
    plt.plot(N, lst)
    plt.title("Error of cg in respect to np, at high dimensions")
    plt.show()


def timings():
    # Generate Systems
    dims = [2, 3, 4, 10, 20, 50, 100, 200, 500, 1000]

    # TIMINGS
    # res1 = []
    # for i, k in enumerate(dims):
    #     print("START N=", k, "\n----------------------------------")
    #     execution_time = 0
    #     c = 0
    #     for j in range(100):
    #         A, b, x_0, sol = genSys(k)
    #         partial_sd = partial(steepest_decent, A0=A, b0=b, x_0=x_0, cap=10000)
    #         try:
    #             execution_time += timeit.timeit(partial_sd, number=1)
    #             c += 1
    #         except:
    #             pass
    #     if c == 0:
    #         print("no valid matrices")
    #         exit()
    #     res1.append(execution_time / c)

    res2 = []
    for i, k in enumerate(dims):
        print("START N=", k, "\n----------------------------------")
        execution_time = 0
        for j in range(10):
            A, b, x_0, sol = genSys(k)
            partial_cg = partial(CG, A0=A, b0=b, x_0=x_0)
            execution_time += timeit.timeit(partial_cg, number=1)
        res2.append(execution_time / 10)

    # Complexity comparisons
    # n1 = dims
    # n2 = [i**2 for i in dims]
    n1 = [i / min(n1) * min(res2) for i in n1]
    # n2 = [i / min(n2) * min(res1) for i in n2]

    # Plot
    # plt.plot(dims, res1)
    plt.plot(dims, res2)
    # plt.plot(dims, n1)
    # plt.plot(dims, n2)

    plt.legend(["steepest _decent", "CG", "n", "n^2"])
    plt.title("log log graph of the complexity")
    plt.xlabel("dimension n of matrix")
    plt.ylabel("time in [s]")
    plt.loglog()

    plt.show()


# SETTING WHICH PARTS TO RUN

# given_system()
# random_system(5)
# error_analysis_dec()
error_analysis_cg()
# timings()
