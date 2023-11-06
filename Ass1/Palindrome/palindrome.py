def main():
    n = 999
    cur = 0
    while n > 99:
        # This saves half the calculations by not repeating the identical other half of the multiplication table
        m = n

        # Stops calculating the current row of n, when the values are smaller than cur, since they are smf.
        while m > 99 and n * m > cur:
            if str(n * m) == str(n * m)[::-1]:
                break

            m = m - 1
        else:
            # If the m loop exits, while still on the first entry, then it is because the first entry is smaller than cur, every value calculated after that will, just by virtue of being lower on the multiplication table be smaller. Exit.

            if n == m:
                return cur

            n = n - 1
            continue

        cur = n * m
        n = n - 1


if __name__ == "__main__":
    print(main())
