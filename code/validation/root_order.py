'''
different orders of the same roots in both denominator
and numerator evaluates to the same sum.
'''

from validation.validation_medthods import random_list_of_nuerator_and_denominator
from validation.validation_medthods import compare_a_list_of_equations_token
from validation.validation_medthods import get_token_expressions
from validation.validation_medthods import test_one_expression
#from validation.validation_medthods import sup_number_of_expressions
from validation.validation_medthods import neural_network_validation, TED_of_list_postfix_eq_as_tokens
from model.equation_interpreter import Equation

import math
import itertools 


#%%

def list_of_test_expressions_with_same_roots(int_roots_only: bool = False, spaceinterval: list = [-5,5], max_num: int = None, max_den: int = None, random_order: list = []):
    # tilføj et max så man max kan få input expressions
    
    if (len(random_order) == 0):
        # greate all options
        num_roots, den_roots = random_list_of_nuerator_and_denominator(spaceinterval, False, int_roots_only)
    else:
        num_roots, den_roots = random_list_of_nuerator_and_denominator(spaceinterval, False, int_roots_only, random_order)
    
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
        y = y[0:(len(y)-1):step_lenght_den]

    
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
    print("combining the num and den")
    combine = itertools.product(all_num_combinations, all_den_combinations)      
    all_combinations = [list(p) for p in combine]    
        
    #all_combinations[0][0].append("/")
    #all_combinations[0][0].append("/")
    
    
    all_combinations = [p[0]+['/']+p[1] for p in all_combinations]
    
    #test_expressions = [list(itertools.chain(*p)) for p in all_combinations]
    
    return all_combinations
#%%
x = list_of_test_expressions_with_same_roots(True, max_num = 5, max_den = 5)



  
#%%
# sum degree
# avage ted
# devision factor
# nuber of fails
# total outputs
avage_TED_pr_sum_d = [[i,0,0,0,0] for i in range(2,18)]


for num_d in range(0,3): #0,8
    for den_d in range(2, 5): #0,10
        for _ in range(0,1): #HUSK AT SÆT OP
            roots_list = list_of_test_expressions_with_same_roots(False, max_num = 5, max_den = 5, random_order = [num_d, den_d])
            nn_out = [neural_network_validation(roots) for roots in roots_list]
            nn_out_valid = []
            for i in range(0, len(nn_out)):
                try:
                    boole = Equation(nn_out[i], "postfix").is_valid()
                except Exception:
                    boole = False
                
                if boole:
                    nn_out_valid.append(nn_out[i])
            avage_TED_pr_sum_d[num_d+den_d-2][3] += len(nn_out)-len(nn_out_valid)
            avage_TED_pr_sum_d[num_d+den_d-2][3] += len(nn_out)
            
            if  len(nn_out_valid) != 0 and len(nn_out_valid) != 1: #gider ikke have den med hvis det er kun er en
                print(f"found multiple!!! {len(nn_out_valid)}")
                ted = TED_of_list_postfix_eq_as_tokens(nn_out_valid)
                avage_TED_pr_sum_d[num_d+den_d-2][1] += ted
                if avage_TED_pr_sum_d[num_d+den_d-2][2] != 0:
                    avage_TED_pr_sum_d[num_d+den_d-2][1] /= 2
                avage_TED_pr_sum_d[num_d+den_d-2][2] += 1
            
            