from functions import *
import sympy as sp

from sympy.printing.mathematica import mathematica_code

#Initialize
mean_terms = 4  #Mean number of terms in each expression, poisson-distributed
mean_arg_len = 1 #Mean number of terms in each argument for composite functions, poisson-distributed
mean_complexity = 1 #Mean number of composite functions in each term, poisson-distributed
sample_size = 10  #Sample space

#Poisson-distributed length
num_terms_pois = 1+np.random.poisson(lam=mean_terms,size=sample_size) #Add one to ensure length>0
sympy_expr_pois = generate_expression(num_terms_pois,mean_complexity,mean_arg_len,sample_size)

sympy_diff_expr_pois = []

mathematica_expr_pois = []

x = sp.symbols('x')

for i in range(sample_size):
    sympy_diff_expr_pois.append(sp.diff(sympy_expr_pois[i],x))
    mathematica_expr_pois.append(mathematica_code(sympy_expr_pois[i]))

### Save data to txt.files
with open('functions_folder/poisson_length/mathematica_expr_pois.txt','w') as file:
    for expr in mathematica_expr_pois:
        file.write(expr + "\n")

with open('functions_folder/poisson_length/sympy_expr_pois.txt','w') as file:
    for expr in sympy_expr_pois:
        file.write(str(expr) + "\n")