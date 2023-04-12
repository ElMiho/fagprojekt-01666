'''
Use this file for methods/functions that can be used in multiple validation test.
'''


from model.tokenize_input import token_input_space
from model.tokenize_input import all_poly
from main import test_an_expression

import random
import torch



def random_list_of_nuerator_and_denominator(concatenate: bool = True):
    '''
    geneate a random sum to be evaluated

    Returns
    -------
    num_roots : List
        all the numerator roots in a list
    den_roots : List
        all the denominator roots in a list

    '''
    
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
     
    if concatenate:
        num_roots.append("/")
        
        return num_roots + den_roots
    
    
    return num_roots, den_roots
