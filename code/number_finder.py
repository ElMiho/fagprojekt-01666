import mpmath

mpmath.dps = 50


print(mpmath.psi(1, 2/3))

key = round(mpmath.psi(1, 2/3), 7)

for z1 in range(1,1000):
    for z2 in range(1,1000):
        conjecture = round(mpmath.pi * mpmath.sqrt(z1) / z2, 7)
        if conjecture == key:
            print(f"z1 = {z1}, z2 = {z2}, {conjecture}")
        







