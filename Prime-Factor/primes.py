a = 600851475143

i = 2
while i**2 <= a:
    if a % i:
        i = i + 1
    else:
        a = a / i
print(a)
