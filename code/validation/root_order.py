'''
different orders of the same roots in both denominator
and numerator evaluates to the same sum.
'''

from validation.validation_medthods import random_list_of_nuerator_and_denominator
from validation.validation_medthods import compare_a_list_of_equations_token

#henter alt fra main som igonre print
from main import test_an_expression

import itertools 




def list_of_test_expressions_with_same_roots():
    # greate all options
    num_roots, den_roots = random_list_of_nuerator_and_denominator(False)
    
    # gets all posible combinations of denominator and numerator
    all_num_combinations = [list(p) for p in itertools.permutations(num_roots)]
    all_den_combinations = [list(p) for p in itertools.permutations(den_roots)]
    
    
    all_combinations = [list(p) for p in itertools.product(all_num_combinations, all_den_combinations)]
    all_combinations[0][0].append("/")
    
    test_expressions = [list(itertools.chain(*p)) for p in all_combinations]
    
    return test_expressions



A = list_of_test_expressions_with_same_roots()


  
a = compare_a_list_of_equations_token(["TT_int", "TT_int","TT_notINT"])
      