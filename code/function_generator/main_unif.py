from functions import *
import sympy as sp

from sympy.printing.mathematica import mathematica_code

#Initialize
max_term_len = 10 #Maximum number of terms in expression, for uniform-distributed lenghts
mean_terms = 4  #Mean number of terms in each expression, poisson-distributed
mean_arg_len = 1 #Mean number of terms in each argument for composite functions, poisson-distributed
mean_complexity = 1 #Mean number of composite functions in each term, poisson-distributed
sample_size = 10  #Sample size

#Uniformly distributed lenght
num_terms_unif = np.random.randint(1, max_term_len, size=sample_size)

sympy_expr_unif = generate_expression(num_terms_unif,mean_complexity,mean_arg_len,sample_size)

sympy_diff_expr_unif = []
mathematica_expr_unif = []

x = sp.symbols('x')

for i in range(sample_size):
    sympy_diff_expr_unif.append(sp.diff(sympy_expr_unif[i],x))
    mathematica_expr_unif.append(mathematica_code(sympy_expr_unif[i]))

### Save data to txt.files
with open('functions_folder/uniform_length/mathematica_expr_unif.txt','w') as file:
    for expr in mathematica_expr_unif:
        file.write(expr + "\n")

with open('functions_folder/uniform_length/sympy_expr_unif.txt','w') as file:
    for expr in sympy_diff_expr_unif:
        file.write(str(expr) + "\n")