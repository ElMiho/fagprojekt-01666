import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import random
import sympy as sp

fraction_list_num = ["-10", "-9" , "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1","2", "3", "4", "5", "6" ,"7", "8", "9", "10"]
fraction_list_den = ["-10", "-9" , "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0"]


#Functions
def int_tuple_list(n, deg_num, deg_den, base_num, base_den):
    #Split into two unique integers using division with remainder by n and base_num**deg_num
    #n_num = remainder, n_den = quotient
    divisor = (base_num**deg_num)
    quo, rem = divmod(n,divisor)
    n_den = quo
    n_num = rem
    poly_fraction = []

    print(f"n: {n}, ")

    #Generate numerator and denominator independently
    poly_num = integer_to_list(n_num, deg_num, base_num)
    poly_den = integer_to_list(n_den, deg_den, base_den)
    poly_fraction.extend((poly_num,poly_den))
    poly_fraction_out = [item for sublist in poly_fraction for item in sublist]
    return poly_fraction_out

def integer_to_list(n, sum_deg, base):
    # allocates index 0 and 1 to roots in numerator and denominator
    l = []
    for i in range(1, sum_deg+1):
        var = n % base**i
        l.append(var // (base**(i-1))+1)
        n = n - var
    l = np.asarray(l)
    return l

def list_to_roots(list_int, deg_num):
    print(list_int)
    print(deg_num)
    fraction_list_num = ["-10", "-9" , "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1","2", "3", "4", "5", "6" ,"7", "8", "9", "10"]
    fraction_list_den = ["-10", "-9" , "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0"]

    list_num = []
    list_den = []

    idx_num = list_int[0:deg_num]
    idx_den = list_int[deg_num:]
    
    for i in idx_num:
        root = fraction_list_num[i-1]
        list_num.append(root)
    for i in idx_den:
        root = fraction_list_den[i-1]
        list_den.append(root)
    
    return list_num,list_den

# Generate_folder for expressions
folder_name_int = "random_expressions_integers"
folder_name_roots = "random_expressions_roots"
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, folder_name_int)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)
final_directory = os.path.join(current_directory, folder_name_roots)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

#### Import sampling parameters
#44x4 np-array [Deg numerator, Deg denominator, Number of samples, Total sample space]
sample_parameters = np.loadtxt('SampleParametersIntegers.txt', delimiter=',')

####  Initialize sampling
base_num = 21
base_den = 11
#num_categories = sample_parameters.shape[0]    commented while testing
num_categories = 44
for i in range(num_categories):
    print("current: " + str(i))
    deg_num = int(sample_parameters[i,0])
    deg_den = int(sample_parameters[i,1])
    #sample_size = int(sample_parameters[i,2])  commented while testing
    sample_size = 10**6                     
    sample_space = int(sample_parameters[i,3])
    
    print("sample_space: " + str(sample_space))
    print("sample_size: " + str(sample_size))
    samples = [random.randint(1, sample_space) for _ in range(sample_size)]

    #From integer to list and saves file in random expression folder
    integer_expressions = []
    root_list_den = []
    root_list_num = []
    sum_deg = deg_num + deg_den
    for j in range(len(samples)):
        n = samples[j]
        list_int = int_tuple_list(n, deg_num, deg_den, base_num, base_den)
        root_num,root_den = list_to_roots(list_int, deg_num)

        root_list_num.append(root_num)
        root_list_den.append(root_den)
        integer_expressions.append(list_int)

    root_list_den = np.asarray(root_list_den)
    root_list_num = np.asarray(root_list_num)
    integer_expressions = np.asarray(integer_expressions)
    #Save txt. file    
    np.savetxt(f'{folder_name_int}/rand_int_expressions-{deg_num}-{deg_den}.txt',integer_expressions, fmt='%i', delimiter=',')
    np.savetxt(f'{folder_name_roots}/rand_roots_num_expressions-{deg_num}-{deg_den}.txt',root_list_num, fmt='%s', delimiter=',')
    np.savetxt(f'{folder_name_roots}/rand_roots_den_expressions-{deg_num}-{deg_den}.txt',root_list_den, fmt='%s', delimiter=',')
