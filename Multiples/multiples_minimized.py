import numpy as np

a = 3
b = 5

arr = np.array([a**i for i in range(1, 11)])
arr = np.append(arr, np.array([b**i for i in range(1, 11)]))
arr = np.delete(arr, (arr > 1000))
print(arr.sum())
