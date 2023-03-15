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
folder_name = "stratified_expressions"
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

    #Generate partioning of sample space
    partitions = 10
    partioned_sample_space = sample_space // partitions
    partioned_sample_size = sample_size // partitions

    #Sample random for each subspace
    samples = []
    min = 1
    max = partioned_sample_space
    for j in range(partitions):
        next_sample = np.random.randint(min,max,size=partioned_sample_size)
        samples.append(next_sample)
        min = max
        max += partioned_sample_size
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
    titles = f"Stratified sampling: Deg numerator: {deg_num}, Deg denominator: {deg_den}"
    np.savetxt(f'{folder_name}/unique_expressions-{deg_num}-{deg_den}.txt',polynomials, fmt='%i', delimiter=',', header=titles)

# %%
