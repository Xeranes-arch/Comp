import numpy as np

i = 2
p = 1
arr = np.array([1, 2])

limit = 4e6 * 0.618
while i < limit:
    n = i + p
    p = i
    i = n
    arr = np.append(arr, i)

sum = np.delete(arr, np.where(arr % 2)).sum()

print(sum)
