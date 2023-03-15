#%%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
#Functions
def integer_to_list(n, sum_deg, base):
    # allocates index 0 and 1 to roots in numerator and denominator
    l = []
    for i in range(1, sum_deg+1):
        var = n % base**i
        l.append(var // (base**(i-1)))
        n = n - var
    l = np.asarray(l)
    return l

# Generate_folder for expressions
folder_name = "unique_expressions"
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, folder_name)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

#### Import sampling parameters
#44x4 np-array [Deg numerator, Deg denominator, Number of samples, Total sample space]
sample_parameters = np.loadtxt('SampleParameters.txt', delimiter=',')

####  Initialize sampling
modulus = 34
#num_categories = sample_parameters.shape[0]   commented while testing
num_categories = 1  #For testing
for i in range(num_categories):
    deg_num = int(sample_parameters[i,0])     
    deg_den = int(sample_parameters[i,1])      
    #sample_size = int(sample_parameters[i,2])  
    sample_size = 100                          #For testing
    sample_space = 1000                        #For testing
    #sample_space = int(sample_parameters[i,3])

    #Generate stepsize
    step_size = sample_space // sample_size
    min = 1
    max = step_size

    #Take a step and draw uniformly backwards
    samples = []
    for i in range(sample_size):
        next_int = np.random.randint(min,max,size=1)
        samples.append(next_int)
        min = max
        max += step_size
    samples = np.asarray(samples)
    samples = samples.flatten()

    #From integer to list and saves file in random expression folder
    polynomials = []
    sum_deg = deg_num + deg_den
    for j in range(len(samples)):
        n = samples[j]
        list = integer_to_list(n,sum_deg,modulus)
        polynomials.append(list)
    polynomials = np.asarray(polynomials)
    #Save txt. file    
    titles = f"Unique sampling: Deg numerator: {deg_num}, Deg denominator: {deg_den}"
    np.savetxt(f'{folder_name}/unique_expressions-{deg_num}-{deg_den}.txt',polynomials, fmt='%i', delimiter=',', header=titles)

# %%
