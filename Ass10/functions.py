import numpy as np
import matplotlib.pyplot as plt


def func(A, b, x):
    return 0.5 * np.matmul(np.transpose(x), np.matmul(A, x)) - np.matmul(
        np.transpose(x), b
    )


def steepest_descent_vis(A0, b0, x_0, cap=10000):
    """Steepest decent algorithm, that also draws a visualization, but doesn't yet show.
    cap:    cut off point for itterations. At 10000 it is usually converging so slowly, that it just wont reach any accuracy. This can be visualized by
    """
    # Doing the transformation here regardless of whether or not necessary. Should not make a performance difference.
    A1 = np.matmul(np.transpose(A0), A0)
    b1 = np.matmul(np.transpose(A0), b0)

    P0 = []
    for i in np.diag(A1):
        P0.append(1 / i)
    P = np.diag(P0)

    A = np.matmul(np.transpose(P), A1)
    b = np.matmul(np.transpose(P), b1)

    x_n = x_0.copy()
    xs = x_n
    k = 0

    # Algorithm steps
    while True:
        x_o = x_n.copy()
        r_n = b - np.matmul(A, x_n)
        alph = np.matmul(np.transpose(r_n), r_n) / np.matmul(
            np.transpose(r_n), np.matmul(A, r_n)
        )
        alph = alph.item()
        x_n += alph * r_n
        xs = np.hstack((xs, x_n))

        # Convergence criteria. Somewhat arbitrary, chosen to deliver solid results.
        if (
            all(abs(i) <= 0.000001 for i in r_n)
            or max(abs(np.array(x_o) - np.array(x_n))) < 0.000001
        ):
            # Visualizing results. OPTION: show plot on its own before layering with visualization eg.
            plt.plot(
                np.array(xs[0, :]),
                np.array(xs[1, :]),
                label=f"SD with {k} Steps.",
                c="b",
            )
            plt.scatter(
                np.array(xs[0, -1]),
                np.array(xs[1, -1]),
                marker="+",
                c="r",
                label="_nolegend_",
            )
            plt.title((f"Paths of Algorithms taken for dim = {len(b)}."))
            # plt.show()
            break

        # Update of progress at every thousand steps. OPTION: Comment in or out as needed.
        if not k % 1000 and k != 0:
            print(k, "STEPS")
            # plt.scatter(
            #     np.array(xs[0, -1]),
            #     np.array(xs[1, -1]),
            #     marker="+",
            #     c="r",
            #     label="_nolegend_",
            # )
            # plt.title((f"SOLUTION for dim N = {len(b)} with {k} Itterations of SD "))
            # plt.plot(np.array(xs[0, :]), np.array(xs[1, :]),c="b")
            # plt.show()

        # Itterations cap to stop at. Makes quit out noticeable.
        if k == cap:
            plt.plot(
                np.array(xs[0, :]),
                np.array(xs[1, :]),
                label=f"SD with {k} Steps.",
                c="b",
            )
            plt.scatter(
                np.array(xs[0, -1]),
                np.array(xs[1, -1]),
                marker="+",
                c="r",
                label="_nolegend_",
            )
            plt.title(
                f"WARNING! SD did not converge. Iteration Cap of {cap} Steps reached."
            )
            print(
                "------------------------------------------\nITTERATION CAP TRIGGERED FOR SD!\nNo Sultion."
            )
            return x_n, False

        k += 1
    return x_n, True


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
            plt.plot(
                np.array(xs[0, :]),
                np.array(xs[1, :]),
                label=f"CG in {n} Steps.",
                c="orange",
            )
            plt.scatter(
                np.array(xs[0, -1]),
                np.array(xs[1, -1]),
                marker="+",
                c="r",
                label="_nolegend_",
            )
            plt.title((f"Paths of Algorithms taken for dim = {len(b)}."))
            break

        beta = (
            np.matmul(np.transpose(r_nn), r_nn) / np.matmul(np.transpose(r_n), r_n)
        ).item()
        p_n = r_nn + beta * p_n
        r_n = r_nn

    return x_n


def steepest_descent(A0, b0, x_0, cap=10000, ctrl=False):
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


def given_system():
    A = np.array([[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]]).astype(
        float
    )
    b = np.matrix([[1], [2], [3], [4]]).astype(float)
    x_0 = np.array([[0.1], [0.1], [0.1], [0.1]])
    sol = np.array(np.linalg.solve(A, b))
    return [A, b, x_0, sol]


def genSys(dim=3):
    A = (np.random.randint(-9, 9, size=(dim, dim))).astype(float)
    while np.linalg.matrix_rank(A) < dim:
        A = (np.random.randint(-9, 9, size=(dim, dim))).astype(float)
    b = np.matrix(np.random.randint(-9, 9, size=(dim, 1))).astype(float)
    x_0 = np.array(np.matrix(np.ones((dim, 1))).astype(float))
    sol = np.array(np.linalg.solve(A, b))
    return A, b, x_0, sol


def steepest_descent_animation(A0, b0, x_0, cap=10000, ctrl=False):
    A = np.matmul(A0.T, A0)
    b = np.matmul(A0.T, b0)
    x_n = x_0.copy()

    history = [x_n.copy()]
    k = 0
    while True:
        x_o = x_n.copy()

        r_n = b - np.matmul(A, x_n)
        alph = np.matmul(np.transpose(r_n), r_n) / np.matmul(
            np.transpose(r_n), np.matmul(A, r_n)
        )
        alph = alph.item()
        x_n += alph * r_n
        history.append(x_n.copy())
        if (
            all(abs(i) <= 0.000001 for i in r_n)
            or max(abs(np.array(x_o) - np.array(x_n))) < 0.000001
        ):
            break
        if k == cap:
            if ctrl:
                raise RuntimeError
            break
        if not k % 1000:
            print(k, "itterations")
        k += 1
    return x_n, history
