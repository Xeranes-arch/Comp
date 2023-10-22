import numpy as np


def check_pal(n):
    a = str(n)
    b = a[::-1]
    if a == b:
        return True
    else:
        return False


n = 999
list = []
while n > 99:
    m = n
    while m > 99:
        if check_pal(n * m):
            break
        m = m - 1
    else:
        n = n - 1
        continue
    list.append((n, m))
    n = n - 1

nr = []
for i in list:
    c, d = i
    nr.append(c * d)

print(np.argsort(nr))
print(len(list))
list = list[np.argsort(nr)]
print(list[0])
