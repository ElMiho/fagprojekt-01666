'''
Use this file for methods/functions that can be used in multiple validation test.
'''

import random
import sys

# SET PATH FOR OSX USERS
if sys.platform == 'darwin':
    sys.path.append('../code')
    
LOAD_MAIN_FLAG = False
if LOAD_MAIN_FLAG:
    from main import test_an_expression
    
from model.tokenize_input import token_input_space
from model.tokenize_input import all_poly
from model.equation_interpreter import Equation
#from model.equation_interpreter import 
from model.tokens import TOKEN_TYPE_ANSWERS


from validation.mathematica_from_python import input_to_lists
from validation.mathematica_from_python import evaluate_sum
from validation.mathematica_from_python import close_session



from data_analysis.int_data.generate_plot import parse_line


def test_one_expression(test_expression, as_string: bool = True):
    return test_an_expression(test_expression, not as_string)

def get_token_expressions(test_expression: list):
    return [test_an_expression(i) for i in test_expression]


def compare_a_list_of_equations_token(equations: list):
    # Vil hvike hvis man sammenligerne token række følgen men skal opdateres når vi kan sammenligne 2 quations med .getMathemetaicalNotation()
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
        
        return [num_roots + den_roots]
    
    
    return num_roots, den_roots



def extend_sum(eq_list, space = [-5,5], int_roots_only = False):
    '''Extends a sum in both denominator and numinator with a random number
    (The sum remains the same).
    '''
    if len(eq_list) > 1:
        eq_list = eq_list[0] +['/']+ eq_list[1]
    index = eq_list.index('/')
    all_eq = []
    if int_roots_only:
        random_number = token_input_space(space[0], space[1], "numinator_int_only")
        
    else:
        random_number = token_input_space(space[0], space[1], "numinator_only")

    for num in random_number:
        all_eq.append(eq_list[:index] + [num] + eq_list[index:] + [num])
    return all_eq


def evaluate_tokenized_sum(test_expression: list):
    
    for expression in test_expression:
        print(expression)
        numerator_roots, denominator_roots = input_to_lists(expression)
        print(evaluate_sum(numerator_roots, denominator_roots))
    close_session()    

     
def find_10_simpelest_evaluations(filepath: str):
    #Vil også gerne have polynomiet
    def max_and_idx(aList):
        maxs = aList[0]
        idx = 0
        for i in enumerate(aList):
            if maxs < i[1]:
                maxs = i[1]
                idx = i[0]
        return maxs, idx
    
    def min_and_idx(aList):
        mins = aList[0]
        idx = 0
        for i in enumerate(aList):
            if mins > i[1]:
                mins = i[1]
                idx = i[0]
        return mins, idx
    
    
    #færdig gør denne kode til at finde de simpelste svar (oversæt dem til tokens og tæl ikke int tokens og ral tokens)
    with open(filepath, "r") as file:
        lines = file.readlines()
    
    
    f10se = [" " for _ in range(10)]
    non_int_tokens = [-1 for _ in range(10)]
    roots_list = [[] for _ in range(10)]
    
    #Man får meget simple (for simple) udtryk så man kan evt tilføje flere tokens til denne og få nogle lidt mere kompliceret
    tokens_numbers = ["TT_INTEGER", "TT_RATIONAL", "TT_ZERO", "TT_ONE"]
    
    for line in lines:
        time, answer, sum_degree, roots = parse_line(line)
        x = 0
        equation = Equation.makeEquationFromString(answer).tokenized_equation
        if equation == []:
            continue
            
        else:
            for token in equation:
                if token.t_type not in tokens_numbers:
                    x += 1
                
            min_val, min_idx = min_and_idx(non_int_tokens)
            max_val, max_idx = max_and_idx(non_int_tokens)
            if min_val == -1:
                f10se[min_idx] = answer
                non_int_tokens[min_idx] = x
                roots_list[min_idx] = [roots]
                
            elif x < max_val:
                f10se[max_idx] = answer
                non_int_tokens[max_idx] = x
                roots_list[max_idx] = [roots]
           
        
    return f10se, non_int_tokens, roots
    



if __name__ == '__main__':
    
    megafile1 = "data_analysis/int_data/megafile.txt"
    megafile2 = "data_analysis/new_rational_data/megafile2_txt"
    f10se, non_int_tokens, roots = find_10_simpelest_evaluations(megafile2)
    
    
    '''
    #evaluate_tokenized_sum(random_list_of_nuerator_and_denominator([-5,5], int_roots_only = True))
    sums = random_list_of_nuerator_and_denominator([-5,5],int_roots_only=True)
    sums2 = ["#", "/", "1/2", "1/5"]
    sums = random_list_of_nuerator_and_denominator([-5,5],int_roots_only=False)
    x = test_one_expression(sums2, False)
    y = test_one_expression(sums2, True)
    #[print(i) for i in x]
    tt_list = x.listOfTokens()
    [print(i) for i in tt_list]
    
    equation = Equation.makeEquationFromString("-Sin(2-EulerGamma)+a/3+(-7/3*2 + Pi^2)-2")
    
    tt_list2 = equation.listOfTokens()
    tt_ = equation.tokenized_equation
    '''
    
  
