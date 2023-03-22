#%%
import numpy as np
def int_tuple_list(n, deg_num, deg_den, base_num, base_den):
    #Split into two unique integers using division with remainder by n and base_num**deg_num
    #n_num = remainder, n_den = quotient
    divisor = (base_num**deg_num)
    quo, rem = divmod(n,divisor)
    n_den = quo
    n_num = rem
    poly_fraction = []
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
    fraction_list_num = ["-5", "-4", "-3", "-(5/2)", "-2", "-(5/3)", "-(3/2)", "-(4/3)", "-(5/4)", "-1", "-(4/5)", "-(3/4)", "-(2/3)", "-(3/5)", "-(1/2)", "-(2/5)", "-(1/3)", "-(1/4)", "-(1/5)", "0", "1/5", "1/4", "1/3", "2/5", "1/2", "3/5", "2/3", "3/4", "4/5", "1", "5/4", "4/3", "3/2", "5/3", "2", "5/2", "3", "4", "5"]
    fraction_list_den = ["-5", "-4", "-3", "-(5/2)", "-2", "-(5/3)", "-(3/2)", "-(4/3)", "-(5/4)", "-1", "-(4/5)", "-(3/4)", "-(2/3)", "-(3/5)", "-(1/2)", "-(2/5)", "-(1/3)", "-(1/4)", "-(1/5)", "0", "1/5", "1/4", "1/3", "2/5", "1/2", "3/5", "2/3", "3/4", "4/5", "5/4", "4/3", "3/2", "5/3", "5/2"]

    math_list_output = []
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
    math_list_output.append((list_num,list_den))
    return math_list_output



roots_expressions = []
n = 12435
deg_num = 2
deg_den = 5
base_den = 34
base_num = 39

list_int = int_tuple_list(n, deg_num, deg_den, base_num, base_den)

roots_expressions = list_to_roots(list_int, deg_num)

print(f"Test expressions\n\n{roots_expressions}")


# %%
