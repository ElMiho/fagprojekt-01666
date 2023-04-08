'''
different orders of the same roots in both denominator
and numerator evaluates to the same sum.
'''

from model.tokenize_input import token_input_space
from model.tokenize_input import input_string_to_tokenize_expression
from model.tokenize_input import all_poly

import random


number_of_test = 1

# denominator and numerator
numerator_roots = token_input_space(-5, 5, "numinator_only")
denominator_roots = token_input_space(-5, 5, "dominator_only")

# all oder over oders, a list of lists
poly_oder_list = all_poly(10)


for _ in range(number_of_test):
    
    # gets a poly oder from the list
    num, den = poly_oder_list[random.randint(0, len(poly_oder_list)-1)]
    
    
    if num != 0:
        
    
    
    


        