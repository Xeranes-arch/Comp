import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from functions import *


# And now for some ridiculous shit.
def run_animation(dim=10, itteration_cap=10001, close=False):
    A, b, x_0, sol = genSys(dim=dim)

    # Example: A simple 2x2 system
    # A = np.array([[4.0, 1.0], [1.0, 3.0]])
    # b = np.array([1.0, 2.0])
    # x_0 = np.array([0.0, 0.0])

    # Set up the figure and axis
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], marker="", linestyle="-", label="Solution Path")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Steepest Descent Animation, dim = {dim}")
    ax.legend()

    # Generator
    def generator(A, b, x_0):
        _, history = steepest_descent_animation(A, b, x_0, cap=itteration_cap)
        for steps, i in enumerate(history):
            flag = False
            for j in [20 * (2**k) for k in np.arange(10)]:
                if steps > j and steps % (j / 5):
                    flag = True
                    break
            if flag:
                continue
            yield i[:2]
        if close:
            plt.close(fig)

    # Initialization function for the animation
    def init():
        line.set_data([], [])
        return (line,)

    # Update function for the animation
    def update(frame):
        x, y = frame[0], frame[1]
        line.set_data(np.append(line.get_xdata(), x), np.append(line.get_ydata(), y))
        ax.scatter(
            [np.linalg.solve(A, b)[0]],
            [np.linalg.solve(A, b)[1]],
            color="r",
            marker="x",
            label="True Solution",
        )

        ax.relim()
        ax.autoscale_view()
        return (line,)

    # Create the steepest descent generator
    gen = generator(A, b, x_0)

    # Create the animation
    animation = FuncAnimation(
        fig,
        update,
        frames=gen,
        init_func=init,
        blit=True,
        repeat=False,
        interval=100,
    )

    # animation.save(
    #     steepest_descent_animation.gif",
    #     writer="ffmpeg",
    # )

    plt.show()
