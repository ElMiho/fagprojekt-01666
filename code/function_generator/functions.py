import numpy as np
import sympy as sp

def init_expression(num_terms,var):
    # Function chooses a number of of basis functions and operators between them
        # Input: 
            # num_terms -> int: Number of terms in expression
            # var -> string: Assigned variable for unary functions in library
        # Output: 
            # expr -> np.array: Generated expression, even indicies are from funtion library, odd indicies from operator library  
    
    # Libraries
    operators_lib = np.asarray(["*", "/", "+", "-", "**"])
    functions_lib = np.asarray([
        "x","C",f"log({var})",
        f"sin({var})",f"cos({var})",
        f"exp({var})",f"sqrt({var})",
        "x**c",f"tan({var})",
        f"acos({var})", f"asin({var})", f"atan({var})",
        f"cosh({var})", f"sinh({var})", f"tanh({var})", 
        f"acosh({var})", f"asinh({var})", f"atanh({var})"
    ])

    #Initialize
    num_operators = num_terms-1
    expr = np.empty(num_terms+num_operators, dtype=object)

    #Assign functions and operators to expression, even indecies --> functions, odd indicies --> operators 
    function_idx = np.random.randint(0, len(functions_lib), size=num_terms)
    operator_idx = np.random.randint(0, len(operators_lib), size=num_operators)
    expr[::2] = functions_lib[function_idx]
    expr[1::2] = operators_lib[operator_idx]
    return expr

def add_parenthesis(expr):
    # Function adds parenthesis to expression at random as nested intervals
        # Input: 
            # expr -> np.array: expression with basis functions and operators, even index are functions odd index are operators
        # Output:
            # expr -> str: expression with parenthesis between each terms
    # Initialize
    right_idx = []
    bounds = []
    last_expr = len(expr)
    bounds.append(last_expr)

    #Constant "counter"
    j=0

    #Loops through function indicies and add integers instead of "C"
    if last_expr>1:   
        for i in range(0,last_expr,2):

            #Change constant to an integer
            if(expr[i]=="C"):
                expr[i] = f"C{j}"
                j+=1

            next_left = i                             #Initialize each function with left-parenthesis
            bound = min([k for k in bounds if k > i]) #Ensures that next right parenthesis does not overlap

            next_right = np.random.randint(i,bound)
            
            next_right += next_right%2                #Ensures that the parenthesis corresponds to a function index.
            right_idx.append(next_right)
            bounds.append(next_right)

            #Add left and right parenthesis to each term
            expr[next_left] = ''.join(["(",expr[next_left]])
            expr[next_right] = ''.join([expr[next_right],")"])
    expr = ''.join(expr)
    return expr

def makeConstantsUnique(expr):
    # Function makes all constants in expression unique
        # Input: 
            # expr -> str: expression with parenthesis as string
        # Output: 
            # expr -> str: expression with unique constants
    idx = expr.find('C')
    i = 0
    while(idx!=-1):
        expr = expr[:idx] + f'K{i}' + expr[idx+1:]
        idx = expr.find('C')
        i+=1
    return expr

def add_argument(expr,mean,var_old,var_new):
    # Function adds argument to each unary function in expression
        # Input: 
            # expr -> str: expression with parenthesis as string
            # mean -> int: mean value for poisson distributed length in argument
            # var_old -> str: current variable in unary function 
            # var_new -> str: variable for the new arguments in unary functions
        # Output: 
            # expr -> str: expression with new arguments in unary variables
    idx = expr.find(var_old)    #Returns -1 if not found
    while(idx!=-1):
        arg_len = 1+int(np.random.poisson(lam=mean,size=1))
        arg = init_expression(arg_len,var_new)
        arg = add_parenthesis(arg)
        # Argument is concatenated to expression
        expr = expr[:idx] + arg + expr[idx+len(var_old):]
        idx = expr.find(var_old)
    return expr

def add_composite(expr,mean_complexity,mean_arg_len):
    # Function adds composite arguments to each unary functions
        # Input: 
            # expr -> str: expression with parenthesis
            # mean_complexity -> int: mean for poisson distributed level of how many composite arguments is added
            # mean_arg_len -> int: mean for poisson distributed length of each composite argument
        # Output:
            # expr -> str: expression with added composite functions to each unary function
    complexity = int(np.random.poisson(lam=mean_complexity,size=1))
    i = 0
    while(i<complexity):    #Recall that each function is initialized with a0 as variable
        expr = add_argument(expr,mean=mean_arg_len,var_old=f'a{i}',var_new=f'a{i+1}')
        i+=1
    expr = expr.replace(f'a{i}','x')
    return expr

def generate_expression(num_terms,mean_complexity,mean_arg_len,sample_size):
    # Function generates a sympy list of generated expressions
        # Input: 
            # num_terms -> int: Number of terms in expression
            # mean_complexity -> int: Mean of poisson distributed 'compositeness'
            # mean_arg_len -> int: Mean of poisson distributed length of composite arguments
            # sample_size -> int: Number of expressions 
        # Output:
            # sympy_expression -> sp.list: list of sympy expressions 
    sympy_expressions = []
    for i in range(sample_size):
        expr = init_expression(num_terms[i],'a0')
        expr = add_parenthesis(expr)
        expr = add_composite(expr,mean_complexity,mean_arg_len)
        expr = sp.parse_expr(expr)
        sympy_expressions.append(expr)
    return sympy_expressions