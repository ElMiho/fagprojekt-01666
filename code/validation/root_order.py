'''
different orders of the same roots in both denominator
and numerator evaluates to the same sum.
'''

from validation.validation_medthods import random_list_of_nuerator_and_denominator
from validation.validation_medthods import compare_a_list_of_equations_token
from validation.validation_medthods import get_token_expressions
#from validation.validation_medthods import sup_number_of_expressions

from model.equation_interpreter import Equation

import itertools 


#%%

def list_of_test_expressions_with_same_roots(int_roots_only: bool = False, spaceinterval: list = [-5,5]):
    # tilføj et max så man max kan få input expressions
    
    
    # greate all options
    num_roots, den_roots = random_list_of_nuerator_and_denominator(spaceinterval, False, int_roots_only)
    
    # gets all posible combinations of denominator and numerator
    all_num_combinations = [list(p) for p in itertools.permutations(num_roots)]
    all_den_combinations = [list(p) for p in itertools.permutations(den_roots)]
    
    
    all_combinations = [list(p) for p in itertools.product(all_num_combinations, all_den_combinations)]
    all_combinations[0][0].append("/")
    
    test_expressions = [list(itertools.chain(*p)) for p in all_combinations]
    
    return test_expressions


x = list_of_test_expressions_with_same_roots(True)


#%%
test_expressions = get_token_expressions(x)

#%%
print(test_expressions[0].getMathmetaicalNotation())

#%%
res = compare_a_list_of_equations_token(x)
  
#%%
    