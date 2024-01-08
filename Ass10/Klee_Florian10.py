import timeit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from functools import partial
from functions import *
from animation import *
from math import log10, floor

np.random.seed()


def visualize_algorithms(problem):
    """Algorithms on random System visualized."""
    A, b, x_0, sol = problem

    max_iterations = 10000

    res1, flag = steepest_descent_vis(A, b, x_0, cap=max_iterations)
    res2 = CG_vis(A, b, x_0)

    if flag:
        print("------------------------------------------\nSD SOLUTION:\n", res1)
        err = max(abs((res1 - sol) / max(sol))) * 100
        print(
            "ERROR of: ",
            float("%.2g" % err[0]),
            "%\n------------------------------------------",
        )
    else:
        plt.title(
            f"WARNING! SD did not converge. Iteration Cap of {max_iterations} Steps reached."
        )

    print("------------------------------------------\nCG SOLUTION:\n", res2)
    err = max(abs((res2 - sol) / max(sol))) * 100
    print(
        "ERROR of: ",
        float("%.2g" % err[0]),
        "%\n------------------------------------------",
    )

    plt.scatter(sol[0], sol[1], c="g", marker="+", label="correct solution")
    plt.legend()
    plt.show()


def timings():
    # Generate Systems OPTION: choose different dims (tends to get expensive time wise real quick)
    dims = [2, 3, 4, 5, 10, 20, 35]
    # dims = np.arange(2, 30, 5)

    # TIMINGS
    rate = []
    res_sd = []
    for i in dims:
        print("START dim =", i)
        execution_time = 0
        successes = 0
        fails = 0
        required_successes = 10
        # makes sure only the matrices that converge adequately are counted
        while True:
            A, b, x_0, sol = genSys(i)
            partial_sd = partial(
                steepest_descent, A0=A, b0=b, x_0=x_0, cap=10000, ctrl=True
            )
            try:
                execution_time += timeit.timeit(partial_sd, number=1)
                successes += 1
            except:
                fails += 1
                pass
            if successes == required_successes:
                rate.append(
                    round((required_successes / (required_successes + fails) * 100), 2)
                )
                break
            if fails == 100:
                exit(f"below {required_successes} in x tries")
        res_sd.append(execution_time / required_successes)

    res_cg = []
    for i, k in enumerate(dims):
        print("START N =", k, "\n----------------------------------")
        execution_time = 0
        for j in range(10):
            A, b, x_0, sol = genSys(k)
            partial_cg = partial(CG, A0=A, b0=b, x_0=x_0)
            execution_time += timeit.timeit(partial_cg, number=1)
        res_cg.append(execution_time / 100)

    # Complexity comparisons
    n1 = dims
    n2 = [i**3 for i in dims]

    n1 = [i / n1[-1] * res_cg[-1] for i in n1]
    n2 = [i / min(n2) * min(res_sd) for i in n2]
    n3 = [i / n1[-1] * res_sd[-1] for i in n1]

    # Plot
    plt.plot(dims, res_sd)
    plt.plot(dims, res_cg)
    plt.plot(dims, n1, c="g")
    plt.plot(dims, n2, c="r")
    plt.plot(dims, n3, c="g")

    plt.legend(["steepest _decent", "CG", "n", "n^3"])
    plt.title("log log graph of the complexity")
    plt.xlabel("dimension n of matrix")
    plt.ylabel("time in [s]")
    plt.loglog()

    plt.show()

    # plt.plot(dims, rate)
    # plt.show()


# # SETTING WHICH PARTS OF THE CODE TO RUN

print("############# Given_system: #############")
visualize_algorithms(given_system())


# OPTION: Set dimension of the matrix that you want to randomly generate and solve.
print("############# Random_system: #############")
dim = 10
visualize_algorithms(genSys(dim))


print("############# Timings: #############")
timings()

# This one is a bit mad and only fun at high dimensions. I wanted to go for cg as a nice comparison too, but I've already put waaay too much effort into this.
# With some specific nice patterns handpicked. The last one that runs and auto closes will run forever. Just saying. It's pretty imo. You can have that shit open all day if you want.

np.random.seed(42)
run_animation(dim=1000, itteration_cap=10001)
np.random.seed(86)
run_animation(dim=1000, itteration_cap=10001)
np.random.seed(42)
run_animation(dim=9, itteration_cap=10001)
np.random.seed(44)
run_animation(dim=10, itteration_cap=10001)
np.random.seed(44)
run_animation(dim=19, itteration_cap=10001)

np.random.seed()
i = 0
while True:
    i += 5
    run_animation(dim=i, itteration_cap=10001, close=True)
