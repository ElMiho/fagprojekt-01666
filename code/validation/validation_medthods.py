'''
Use this file for methods/functions that can be used in multiple validation test.
'''


from model.tokenize_input import token_input_space
from model.tokenize_input import all_poly
from model.equation_interpreter import Equation

from main import test_an_expression

import random




def get_token_expressions(test_expression: list):
    return [test_an_expression(i) for i in test_expression]

def sup_number_of_expressions(max_ex: int, test_expression: list):
    length = len(test_expression)
    print(length)
    if max_ex < length:
        t_e = []
        steps = length // max_ex
        for i in range(0, steps-1):
            print(i)
            t_e.append(test_expression[i*steps])
        return t_e
    return test_expression
    
def compare_a_list_of_equations_token(equations: list):
    n = len(equations)
    count = 0
    total = 0
    
    for i in range(n-1):
        eq = equations[i]
        
        for j in range(i + 1, n):
            total += 1
            if eq == equations[j]:
                count += 1
                
    
    return count/total

def random_list_of_nuerator_and_denominator(spaceinterval: list = [-5,5] ,concatenate: bool = True, int_roots_only: bool = False):
    '''
    geneate a random sum to be evaluated

    Returns
    -------
    num_roots : List
        all the numerator roots in a list
    den_roots : List
        all the denominator roots in a list

    '''
    if (int_roots_only):
        roots_num_kind = "numinator_int_only"
        roots_den_kind = "dominator_int_only"
    else :
        roots_num_kind = "numinator_only"
        roots_den_kind = "dominator_only"
    
    # denominator and numerator
    numerator_roots = token_input_space(spaceinterval[0], spaceinterval[1], roots_num_kind)
    denominator_roots = token_input_space(spaceinterval[0], spaceinterval[1], roots_den_kind)
    
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

