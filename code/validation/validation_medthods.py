'''
Use this file for methods/functions that can be used in multiple validation test.
'''
import torch
import random
import sys

# SET PATH FOR OSX USERS
if sys.platform == 'darwin':
    sys.path.append('../code')
    
LOAD_MAIN_FLAG = False
if LOAD_MAIN_FLAG:
    from main import test_an_expression
 


from GPT2_Inference_simple import neural_network
    
from model.tokenize_input import token_input_space
from model.tokenize_input import all_poly
from model.equation_interpreter import Equation
#from model.equation_interpreter import 
from model.tokens import TOKEN_TYPE_ANSWERS

from model import equation_interpreter
#from validation.mathematica_from_python import input_to_lists
#from validation.mathematica_from_python import evaluate_sum
#from validation.mathematica_from_python import close_session
from model.tokens import Token
from validation.TED import graph_from_postfix, TreeEditDistance

from model.vocabulary import vocabulary_answers as target_vocabulary
from model.vocabulary import vocabulary_expressions as source_vocabulary

from data_analysis.int_data.generate_plot import parse_line
from validation.postfix_tokens_to_tree import generate_nodes_from_postfix
from validation.tree_edit_distance import Node, Tree, tree_edit_distance, plot_graph




def neural_network_validation(roots: list):
    output = neural_network(roots)
    return output[len(roots)+2:len(output)-1]

def infix_equation_to_posfix():
    #skriv kode
    return None

def test_one_expression(test_expression, as_string: bool = True):
    #OLD MODEL
    return test_an_expression(test_expression, not as_string)

def get_token_expressions(test_expression: list):
    return [test_an_expression(i) for i in test_expression]

def TED_of_list_postfix_eq_as_tokens(equations: list):
    n = len(equations)
    ted = 0
    total = 0
    trees = token_list_to_trees(equations) 
    for i in range(n-1):
        Ti = trees[i]
        for j in range(i + 1, n):
            total += 1
            dist = TreeEditDistance().calculate(trees[j], Ti)
            ted += dist[0]    
    if total == 0:
        return 0
    return ted/total

def token_list_to_trees(tokens: list):
    trees = []
    for eq in tokens:
        _, T = graph_from_postfix(eq)
        trees.append(T)
    return trees

def TED_of_list_postfix_eq_as_tokens_old(equations: list):
    n = len(equations)
    ted = 0
    total = 0
    trees = token_list_to_trees(equations)
    for i in range(n-1):
        Ti = trees[i]
        for j in range(i + 1, n):
            total += 1
            treedist, operations, forestdist_dict = tree_edit_distance(Ti, trees[j])
            ted += treedist[-1,-1]
    
    if total == 0:
        return 0
    return ted/total
            
            
def token_list_to_trees_old(tokens: list):
    trees = []
    for eq in tokens:
        nodes = generate_nodes_from_postfix(eq)
        T = Tree(nodes = nodes, root = nodes[0])
        trees.append(T)
    return trees
    
    
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

def random_list_of_nuerator_and_denominator(spaceinterval: list = [-5,5] ,concatenate: bool = True, int_roots_only: bool = False, random_order: list = []):
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
    
    if (len(random_order) == 0):
    # all oder over oders, a list of lists
        poly_oder_list = all_poly(10)
        # gets a poly oder from the list
        num, den = poly_oder_list[random.randint(0, len(poly_oder_list)-1)]
    else:
       num = random_order[0]    
       den = random_order[1]
    
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



def extend_sum(eq_list, space = [-5,5], int_roots_only = False):
    '''Extends a sum in both denominator and numinator with a random number
    (The sum remains the same).
    '''
    #if len(eq_list) > 1:
        #eq_list = eq_list[0] +['/']+ eq_list[1]
    index = eq_list.index('/')
    all_eq = []
    if int_roots_only:
        random_number = token_input_space(space[0], space[1], "numinator_int_only")
        
    else:
        random_number = token_input_space(space[0], space[1], "numinator_only")

    for num in random_number:
        
        all_eq.append(eq_list[:index] + [str(num)] + eq_list[index:] + [str(num)])
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
    
def to_indices(scores):
    indices = torch.max(scores, dim=1)
    return indices

def sentence_from_indices(indices, vocab, strict=True):
    out = []
    for index in indices:
        index = index.item()
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            return " ".join(out)
        else:
            out.append(vocab.getToken(index))
    return " ".join(out)

def input_roots_num_den(roots):
    idx = roots.index("/")
    if roots[0] == "#":
        return idx-1, len(roots)-idx-1
    return idx, len(roots)-idx-1

def roots_to_strings(roots):
    return [str(root) for root in roots] 

def valid_equation(tokens):
    try:
        return Equation(tokens, "postfix").is_valid()
    except Exception:
        return
    
def posible_degrees(den_max):
    deg = []
    for i in range(0, den_max-1):
        for j in range(i+2, den_max+1):
            deg.append([i,j])
    return deg
        
if __name__ == '__main__':
    
   
    
    '''
    megafile1 = "data_analysis/int_data/megafile.txt"
    megafile2 = "data_analysis/new_rational_data/megafile2_txt"
    f10se, int_tokens, roots_int = find_10_simpelest_evaluations(megafile1)
    f10seM2, non_int_tokens, roots = find_10_simpelest_evaluations(megafile2)
    '''
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
    
    '''
    #Til rapport ilustration
    equation = Equation.makeEquationFromString("1/9*(-4+3*Pi^2-4*Log(4))")
    equation.convertToPostfix()
    tokens = equation.tokenized_equation
    
    #output 
    math = [token.t_type for token in tokens]
    tokens = " ".join(math)
    token_output = [Token(token) for token in math]
    equation_no_values = Equation(token_output ,"postfix")
    print(equation_no_values.getMathemetaicalNotation())
    token_idxes = [target_vocabulary.getIndex(token.t_type) for token in token_output]
    
    #input
    tt_input = ["#","/","-1","1/2","1/2"]
    token_idxes2 = [source_vocabulary.getIndex(token) for token in tt_input]
    '''
    
    
    list_of_token_lists = []
    equation = equation_interpreter.Equation.makeEquationFromString("(Log(7)-Pi)*2")
    equation.convertToPostfix()
    tokens = equation.tokenized_equation
    #print(tokens)
    #[print(target_vocabulary.getIndex(token.t_type)) for token in tokens]
    list_of_token_lists.append(tokens)
    
    equation = equation_interpreter.Equation.makeEquationFromString("2*(Log(7)-Pi)")
    equation.convertToPostfix()
    tokens = equation.tokenized_equation
    list_of_token_lists.append(tokens)
    
    equation = equation_interpreter.Equation.makeEquationFromString("(Log(7)+Pi)*2")
    equation.convertToPostfix()
    tokens = equation.tokenized_equation
    list_of_token_lists.append(tokens)
    print(TED_of_list_postfix_eq_as_tokens(list_of_token_lists))
    
    
    #output = neural_network_validation(["#","/","0","0","0"])