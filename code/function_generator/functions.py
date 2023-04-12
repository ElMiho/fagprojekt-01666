import numpy as np
import sympy as sp

def init_expression(num_terms,arg):
    ### Functions and operators
    operators_lib = np.asarray(["*", "/", "+", "-", "**"])
    functions_lib = np.asarray(["x","C",f"log({arg})",f"sin({arg})",f"cos({arg})",f"exp({arg})",f"sqrt({arg})","x**c",f"tan({arg})"])

    num_operators = num_terms-1

    #Initialize
    expr = np.empty(num_terms+num_operators, dtype=object)

    #Assign functions and operators to expression, even indecies --> functions, odd indecies --> operators 
    function_idx = np.random.randint(0, len(functions_lib), size=num_terms)
    operator_idx = np.random.randint(0, len(operators_lib), size=num_operators)
    expr[::2] = functions_lib[function_idx]
    expr[1::2] = operators_lib[operator_idx]
    return expr

def add_parenthesis(expr):
    #Add parenthesis to expression randomly
    #Args: expressions -> even index are functions odd index are operators
    # Function create parenthesis at random as nested intervals
    # Returns an expression with randomized parenthesis around each term
    # Return is a string

    # Initialize
    left_idx = []
    right_idx = []
    bounds = []
    last_expr = len(expr)
    bounds.append(last_expr)

    if last_expr>1:   
        #Note that expression are on even indicies
        for i in range(0,last_expr,2):
            next_left = i
            left_idx.append(next_left)
            bound = min([k for k in bounds if k > i])

            #Different distributions
            next_right = np.random.randint(i,bound)
            
            next_right += next_right%2          #Makes sure that the parenthesis is at an even index corresponding to a function index.
            right_idx.append(next_right)
            bounds.append(next_right)

            #Add left and right parenthesis to each term
            expr[next_left] = ''.join(["(",expr[next_left]])
            expr[next_right] = ''.join([expr[next_right],")"])
    expr = ''.join(expr)
    return expr

def add_argument(expr,mean,var_old,var_new):
    #To do: Find the elements in the given array that are equal to a
    #Input: Numpy array of strings, with 'a' as inputs, mean_complexity: 
    #Return: Numpy array of strings as functions in x.
    idx = expr.find(var_old)
    while(idx!=-1):
        arg_len = 1+int(np.random.poisson(lam=mean,size=1))
        arg = init_expression(arg_len,var_new)
        arg = add_parenthesis(arg)
        # Add argument is concatenated to expression
        expr = expr[:idx] + arg + expr[idx+len(var_old):]
        idx = expr.find(var_old)
    return expr

def add_composite(expr,mean_complexity,mean_arg_len):
    complexity = int(np.random.poisson(lam=mean_complexity,size=1))
    i = 0
    while(i<complexity):
        expr = add_argument(expr,mean=mean_arg_len,var_old=f'a{i}',var_new=f'a{i+1}')
        i+=1
    expr = expr.replace(f'a{i}','x')
    return expr

def generate_expression(num_terms,mean_complexity,mean_arg_len,sample_size):
    sympy_expressions = []
    for i in range(sample_size):
        expr = init_expression(num_terms[i],'a0')
        expr = add_parenthesis(expr)
        expr = add_composite(expr,mean_complexity,mean_arg_len)
        expr = sp.parse_expr(expr)
        sympy_expressions.append(expr)
    return sympy_expressions