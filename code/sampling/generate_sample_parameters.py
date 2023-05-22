import sys
sys.path.append('../')

from model.equation_interpreter import Equation
import numpy as np
import linecache
from scipy.io import savemat

def save_data(file_names):
#Returns the abortion_rate, the length of each line in each sample and sample size_factors
#Saves them as matlab files.
    token_lengths = []
    abortion_rate = []
    sample_size = []

    for string in file_names:
        [a, b] = string.split("-")
        a = int(a)
        b = int(b)
        # input_file =  f"../data_generation/7_sec_data/data/answers-{a}-{b}.txt"
        input_file = f"./abortion_rate_int/{a}-{b}.txt"
        number_of_lines = sum(1 for i in open(input_file, 'rb'))
        num_non_zeros = number_of_lines

        line_length = []
        token_length = 0

        for line_index in range(1,number_of_lines+1):
            raw_equation = linecache.getline(input_file, line_index)
            equation = Equation.makeEquationFromString(raw_equation)
            tokenized_length = len(equation.tokenized_equation)

            if tokenized_length == 0:
                num_non_zeros -= 1

            token_length += tokenized_length
            line_length.append(tokenized_length)

        token_lengths.append(line_length)
        abortion_rate.append((a,b,1-(num_non_zeros/number_of_lines)))
        sample_size.append((a,b,1/(num_non_zeros/number_of_lines)))

    token_lengths = np.asarray(token_lengths)
    abortion_rate = np.asarray(abortion_rate)
    sample_size = np.asarray(sample_size)
    return token_lengths,abortion_rate,sample_size

# file_names = np.loadtxt("file_names.txt")
file_names = [
    "0-2", "0-3", "0-4", "0-5", "0-6", "0-7", "0-8", "0-9", "0-10", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8", "1-9", "1-10", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9", "2-10", "3-5", "3-6", "3-7", "3-8", "3-9", "3-10", "4-6", "4-7", "4-8", "4-9", "4-10", "5-7", "5-8", "5-9", "5-10", "6-8", "6-9", "6-10", "7-9", "7-10", "8-10"
]

token_lengths,abortion_rate,sample_size = save_data(file_names)
print(f"test{sample_size}")
#Export data as matlab file
savemat('abort_rate_integer.mat',{'abortion_rate': abortion_rate})
savemat('Token_Length_integer.mat',{'Token_len': token_lengths})
savemat('Sample_size_integer.mat',{'sample_size': sample_size})

# Generate 44x3 array with [Deg numerator, Deg denominator, Number of samples, Total sample space]

sample_parameters = []

number_of_tokens_num = 21
number_of_tokens_den = 11
total_samples = 40*10**6
categories = 44
n = total_samples//categories
for i in range(categories):
    num = sample_size[i,0]
    den = sample_size[i,1]
    sample_factor = sample_size[i,2]
    number_of_samples = n*sample_factor
    omega = number_of_tokens_num**num*number_of_tokens_den**den     #Change according to different bases
    sample_parameters.append((num,den,number_of_samples,omega))
sample_parameters = np.asarray(sample_parameters)
titles = "Deg numerator, Deg denominator, Number of samples, Total sample space"
np.savetxt('SampleParametersIntegers.txt',sample_parameters, fmt='%i', delimiter=',', header=titles)
