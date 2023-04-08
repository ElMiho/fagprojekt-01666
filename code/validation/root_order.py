'''
different orders of the same roots in both denominator
and numerator evaluates to the same sum.
'''

from model.tokenize_input import token_input_space
from model.tokenize_input import all_poly

import random
import itertools 



def random_list_of_nuerator_and_denominator():
    # denominator and numerator
    numerator_roots = token_input_space(-5, 5, "numinator_only")
    denominator_roots = token_input_space(-5, 5, "dominator_only")
    
    # all oder over oders, a list of lists
    poly_oder_list = all_poly(10)
    
    
    # gets a poly oder from the list
    num, den = poly_oder_list[random.randint(0, len(poly_oder_list)-1)]
    
    num_roots = []
    den_roots = []
    
    if num == 0:
        num_roots.append("#")
    else:
        for _ in range(num):
             num_roots.append(str(numerator_roots[random.randint(0, len(numerator_roots)-1)]))
             
    for _ in range(den):
        den_roots.append(str(denominator_roots[random.randint(0, len(denominator_roots)-1)]))
        
    return num_roots, den_roots


def list_of_test_expressions_with_same_roots():
    # greate all options
    num_roots, den_roots = random_list_of_nuerator_and_denominator()
    
    # gets all posible combinations of denominator and numerator
    all_num_combinations = [list(p) for p in itertools.permutations(num_roots)]
    all_den_combinations = [list(p) for p in itertools.permutations(den_roots)]
    
    
    all_combinations = [list(p) for p in itertools.product(all_num_combinations, all_den_combinations)]
    all_combinations[0][0].append("/")
    
    test_expressions = [list(itertools.chain(*p)) for p in all_combinations]
    
    return test_expressions


    
    
    
   
    



        