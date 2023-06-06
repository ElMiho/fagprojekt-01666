'''
different orders of the same roots in both denominator
and numerator evaluates to the same sum.
'''

from validation.validation_medthods import random_list_of_nuerator_and_denominator
from validation.validation_medthods import compare_a_list_of_equations_token
from validation.validation_medthods import get_token_expressions
from validation.validation_medthods import test_one_expression
#from validation.validation_medthods import sup_number_of_expressions

from model.equation_interpreter import Equation

import math
import itertools 


#%%

def list_of_test_expressions_with_same_roots(int_roots_only: bool = False, spaceinterval: list = [-5,5], max_num: int = None, max_den: int = None):
    # tilføj et max så man max kan få input expressions
    
    
    # greate all options
    num_roots, den_roots = random_list_of_nuerator_and_denominator(spaceinterval, False, int_roots_only)
    
    print(f"The random poly is:\n {num_roots} / {den_roots}")
    
    num_roots_len = len(num_roots)
    den_roots_len = len(den_roots)
        
    x = list(itertools.permutations(num_roots))
    y = list(itertools.permutations(den_roots))
    
    step_lenght_num = math.factorial(num_roots_len)//max_num
    step_lenght_den = math.factorial(den_roots_len)//max_den
    
    
    if max_num < math.factorial(num_roots_len) and step_lenght_num != 0:
        x = x[0:(len(x)-1):step_lenght_num]
    
    if max_den < math.factorial(den_roots_len) and step_lenght_den != 0:
        y = y[0:(len(x)-1):step_lenght_den]

    
    """
    # gets all posible combinations of denominator and numerator
    all_num_combinations = [list(p) for i,p in enumerate(x) if i%step_lenght_num == 0]
    all_den_combinations = [list(p) for i,p in enumerate(y) if i%step_lenght_den == 0]
    """
    
    # gets all posible combinations of denominator and numerator
    print("calcutating all num combinations..")
    all_num_combinations = [list(p) for p in x]
    print("calcutating all den combinations..")
    all_den_combinations = [list(p) for p in y]
    
    """
    #lav en liste med alle idx af all:num og all_den
    len_num_com = [1 for _ in all_num_combinations]
    len_den_com = [1 for _ in all_den_combinations]
    """
   
    ## denne her der tager lang tid
    combine = itertools.product(all_num_combinations, all_den_combinations)      
    all_combinations = [list(p) for p in combine]    
        
    all_combinations[0][0].append("/")
    
    test_expressions = [list(itertools.chain(*p)) for p in all_combinations]
    
    return test_expressions


x = list_of_test_expressions_with_same_roots(True, max_num = 10, max_den = 20)


#%%
test_expressions = get_token_expressions(x)


#%%
print(test_expressions[0].getMathemetaicalNotation())

#%%
res = compare_a_list_of_equations_token(x)
  
#%%
    