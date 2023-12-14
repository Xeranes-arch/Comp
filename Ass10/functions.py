import numpy as np
import matplotlib.pyplot as plt


def func(A, b, x):
    return 0.5 * np.matmul(np.transpose(x), np.matmul(A, x)) - np.matmul(
        np.transpose(x), b
    )


def steepest_decent_vis(A0, b0, x_0, cap=10000):
    A = np.matmul(np.transpose(A0), A0)
    b = np.matmul(np.transpose(A0), b0)

    x_n = x_0.copy()
    xs = x_n
    k = 0
    while True:
        x_o = x_n.copy()
        r_n = b - np.matmul(A, x_n)
        alph = np.matmul(np.transpose(r_n), r_n) / np.matmul(
            np.transpose(r_n), np.matmul(A, r_n)
        )
        alph = alph.item()
        x_n += alph * r_n
        xs = np.hstack((xs, x_n))

        if (
            all(abs(i) <= 0.000001 for i in r_n)
            or max(abs(np.array(x_o) - np.array(x_n))) < 0.000001
        ):
            plt.plot(np.array(xs[0, :]), np.array(xs[1, :]))
            plt.scatter(
                np.array(xs[0, -1]),
                np.array(xs[1, -1]),
                marker="+",
                c="r",
                label="_nolegend_",
            )
            plt.title((f"SOLUTION for N = {len(b)} with {k} STEPS"))
            # plt.show()
            break
        if not k % 1000 and k != 0:
            print(k, "STEPS")
            plt.scatter(
                np.array(xs[0, -1]),
                np.array(xs[1, -1]),
                marker="+",
                c="r",
                label="_nolegend_",
            )
            plt.title((f"SOLUTION for N = {len(b)} with {k} STEPS"))
        if k == cap:
            plt.plot(np.array(xs[0, :]), np.array(xs[1, :]))
            print("cap triggered")
            break

        k += 1
    return x_n


def steepest_decent(A0, b0, x_0, cap=10000, ctrl=False):
    A = np.matmul(A0.T, A0)
    b = np.matmul(A0.T, b0)
    x_n = x_0.copy()

    k = 0
    while True:
        x_o = x_n.copy()

        r_n = b - np.matmul(A, x_n)
        alph = np.matmul(np.transpose(r_n), r_n) / np.matmul(
            np.transpose(r_n), np.matmul(A, r_n)
        )
        alph = alph.item()
        x_n += alph * r_n
        if (
            all(abs(i) <= 0.000001 for i in r_n)
            or max(abs(np.array(x_o) - np.array(x_n))) < 0.000001
        ):
            break
        if k == cap:
            if ctrl:
                raise RuntimeError
            print("cap triggered")
            break
        k += 1
    return x_n


def CG(A0, b0, x_0):
    """This guy performs so well, Error wise, that I'm using it as a measure for the Error of SD."""
    A = np.matmul(np.transpose(A0), A0)
    b = np.matmul(np.transpose(A0), b0)

    r_n = b - np.matmul(A, x_0)
    if all(np.isclose(r_n, 0, atol=0.00001)):
        return x_0

    p_n = r_n
    x_n = x_0.copy()
    n = 0
    while True:
        alph = (
            np.matmul(np.transpose(r_n), r_n)
            / np.matmul(np.transpose(p_n), np.matmul(A, p_n))
        ).item()

        x_n += alph * p_n
        r_nn = r_n - alph * np.matmul(A, p_n)

        beta = (
            np.matmul(np.transpose(r_nn), r_nn) / np.matmul(np.transpose(r_n), r_n)
        ).item()
        p_n = r_nn + beta * p_n
        r_n = r_nn

        if n == len(b):
            break

        n += 1
    return x_n


def CG_vis(A0, b0, x_0):
    A = np.matmul(np.transpose(A0), A0)
    b = np.matmul(np.transpose(A0), b0)

    r_n = b - np.matmul(A, x_0)
    if all(np.isclose(r_n, 0, atol=0.00001)):
        return x_0
    p_n = r_n
    x_n = x_0.copy()
    xs = x_n.copy()
    n = 0
    while True:
        alph = (
            np.matmul(np.transpose(r_n), r_n)
            / np.matmul(np.transpose(p_n), np.matmul(A, p_n))
        ).item()
        x_n += alph * p_n
        n += 1
        xs = np.hstack((xs, x_n))
        r_nn = r_n - alph * np.matmul(A, p_n)
        # if all(np.isclose(r_nn, 0)):
        if n == len(b):
            plt.plot(np.array(xs[0, :]), np.array(xs[1, :]))
            plt.scatter(
                np.array(xs[0, -1]),
                np.array(xs[1, -1]),
                marker="+",
                c="r",
                label="_nolegend_",
            )
            break

        beta = (
            np.matmul(np.transpose(r_nn), r_nn) / np.matmul(np.transpose(r_n), r_n)
        ).item()
        p_n = r_nn + beta * p_n
        r_n = r_nn

    return x_n


def genSys(i):
    A = (np.random.randint(-9, 9, size=(i, i))).astype(float)
    while np.linalg.matrix_rank(A) < i:
        A = (np.random.randint(-9, 9, size=(i, i))).astype(float)
    # A = np.matrix(np.random.randint(-9, 9, size=(i, i))).astype(float)
    b = np.matrix(np.random.randint(-9, 9, size=(i, 1))).astype(float)
    x_0 = np.array(np.matrix(np.ones((i, 1))).astype(float))
    sol = np.array(np.linalg.solve(A, b))
    return A, b, x_0, sol
