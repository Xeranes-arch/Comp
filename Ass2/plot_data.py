import timeit
import numpy as np
import matplotlib.pyplot as plt


def main():
    t = np.loadtxt("t.dat")
    x = np.loadtxt("x.dat")
    u = np.loadtxt("u.dat")

    plot2D(t, x, u)
    plot1D(t, x, u)
    anim(t, u)
    average(u)


def plot2D(t, x, u):
    X, T = np.meshgrid(x, t)

    CS = plt.pcolormesh(X, T, u)
    plt.contour(X, T, u, 5, colors="k")
    plt.colorbar(CS)
    plt.title("Temperature T(x,t)")
    plt.xlabel("position x in [m]")
    plt.ylabel("time t in [s]")
    plt.show()


def plot1D(t, x, u):
    idx = np.absolute(t - 50).argmin()
    plt.plot(x, u[idx, :])
    plt.title("Temperature T(x,50)")
    plt.xlabel("position x in [m]")
    plt.ylabel("time t in [s]")
    plt.show()


def anim(t, u):
    a = np.vsplit(u, 50)

    fig, ax = plt.subplots()
    for i, img in enumerate(a):
        ax.clear()
        ax.imshow(img, vmin=0, vmax=1, extent=[0, 0.1, 0, 0.002])
        ax.set_title(f"T(x,{int(t[i])})")
        ax.set_xlabel("position x in [m]")
        ax.set_yticks([])

        plt.pause(0.1)


def average(u):
    lst = list(u[27])
    print(np.mean(u[27]))
    print(sum(lst) / len(lst))


if __name__ == "__main__":
    main()
