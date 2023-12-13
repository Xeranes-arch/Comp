import timeit
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from functions import *

np.random.seed(420)

# Given System
A = np.matrix([[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]]).astype(
    float
)
b = np.matrix([[1], [2], [3], [4]]).astype(float)
x_0 = np.array([[0.1], [0.1], [0.1], [0.1]])
sol = np.array([[-1.27619048], [1.87619048], [0.57142857], [2.43809524]])


N = 1000

A, b = genSys(N)
x_0s = []
for i in np.arange(10000):
    arr = np.array(np.matrix(np.ones((i, 1))).astype(float))
    x_0s.append(arr)
x_0 = x_0s[N]


# exit()

# Functions on given System
res = steepest_decent_vis(A, b, x_0)
# assert np.isclose(res, sol, atol=0.0001).all()
# print("ERROR of: ", [float((res[i] - sol[i])[0]) for i in range(len(res))])

# err = []
# for i in np.arange(1000, step=10):
#     res = steepest_decent(A, b, x_0, i)
#     err.append(max([float((res[i] - sol[i])[0]) for i in range(len(res))]))

# plt.plot(np.arange(1000, step=10), err)
# plt.show()


res = CG_vis(A, b, x_0)
# assert np.isclose(res, sol, atol=0.0001).all()
# print("ERROR of: ", [float((res[i] - sol[i])[0]) for i in range(len(res))])


exit()

# Generate Systems
xax = (
    list(np.arange(2, 11))
    + list(np.arange(15, 26, 5))
    + list(np.arange(30, 51, 10))
    + list(np.arange(60, 101, 20))
)
x_0s = []
for i in xax:
    arr = np.matrix(np.ones((i, 1))).astype(float)
    x_0s.append(arr)
x_0 = x_0s[i]
# TIMINGS
res1 = []
for i, k in enumerate(xax):
    print("START N=", k, "\n----------------------------------")
    x_0 = x_0s[i]
    execution_time = 0
    for j in range(1):
        A, b = genSys(k)
        partial_sd = partial(steepest_decent, A0=A, b0=b, x_0=x_0)
        execution_time += timeit.timeit(partial_sd, number=1)
    res1.append(execution_time / 1)

res2 = []
for i, k in enumerate(xax):
    print("START N=", k, "\n----------------------------------")
    x_0 = x_0s[i]
    execution_time = 0
    for j in range(10):
        A, b = genSys(k)
        partial_cg = partial(CG, A0=A, b0=b, x_0=x_0)
        execution_time += timeit.timeit(partial_cg, number=1)
    res2.append(execution_time / 10)

# Complexity comparisons
n1 = xax
n2 = [i**2 for i in xax]
n1 = [i / min(n1) * min(res2) for i in n1]
n2 = [i / min(n2) * min(res1) for i in n2]

# Plot
plt.plot(xax, res1)
plt.plot(xax, res2)
plt.plot(xax, n1)
plt.plot(xax, n2)

plt.legend(["steepest _decent", "CG", "n", "n^2"])
plt.title("log log graph of the complexity")
plt.xlabel("dimension n of matrix")
plt.ylabel("time in [s]")
plt.loglog()

plt.show()
