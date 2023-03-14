import math
import numpy as np

with open('SampleSize.txt', 'r') as file:
    # Read the contents of the file and split it into lines
    sample_size = file.read().splitlines()
# Convert the list of lines into a NumPy array
array = np.array(sample_size)

def sample_categori(sample_size, total_needed):
    #Returns a 2D-array with sample size in each category and number of samples from each
    omega = []  
    token_size_num = 39
    token_size_den = 34
    for i in range(44):
        num = sample_size[i,0]
        den = sample_size[i,1]
        omega.append(token_size_num**num*token_size_den**den)
    omega = np.asarray(omega)




print(f"sample_size: {sample_size}\n")

def integer_to_list(n, numerator, denominator, base_numerator, base_denominator):
    # allocates index 0 and 1 to roots in numerator and denominator
    l = [numerator, denominator]

    for i in range(1, numerator):
        var = n % base_numerator**i
        l.append(var // (base_numerator**(i-1)))
        n = n - var

    for i in range(1, denominator):
        var = n % base_denominator**i
        l.append(var // (base_denominator**(i-1)))
        n = n - var
    
    assert len(var) == 2+numerator+denominator
    return l

def list_to_integer(list, base):
    n = 0
    for idx, x in enumerate(list):
        n += x * base**idx
    return n