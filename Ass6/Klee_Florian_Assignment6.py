import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Ok. This is kinda high and takes a while. But *resolution*.
n = 100

# Set up Matrix
main = np.full(n**2, -4)
first = np.ones(n**2 - 1)
first[n - 1 :: n] = 0
third = np.ones(n**2 - n)

A = (
    np.diag(main, 0)
    + np.diag(first, 1)
    + np.diag(first, -1)
    + np.diag(third, n)
    + np.diag(third, -n)
)

# I still don't understand why the vector is of this form...
bottom = -np.ones(n)
sides = np.zeros(n**2 - n)
bounds = np.append(bottom, sides)
v = linalg.solve(A, bounds).reshape(n, n)

# Grid
x = np.linspace(0, 1, n)
y = x.copy()

# Plot
plt.pcolor(x, y, v, vmin=0, vmax=1)
plt.colorbar()

plt.xlabel("x")
plt.ylabel("y")
plt.title("u(x,y)")

plt.show()
