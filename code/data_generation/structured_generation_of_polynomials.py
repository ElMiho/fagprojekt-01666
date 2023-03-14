import math



def integer_to_list(n, numerator, denominator, base):
    l = [numerator, denominator]
    sum_deg = numerator + denominator
    for i in range(1, sum_deg + 1):
        var = n % base**i
        l.append(var // (base**(i-1)))
        n = n - var
    return l

print(integer_to_list(1 + 2 * 45 + 3 * 45**2, 2, 4, 45))

total = 45**6
needed = 10**6
step_size = total // needed

list = []

for i in range(0, total, step_size):
    list.append(
        integer_to_list(i, 6, 45)
    )

print(len(list))