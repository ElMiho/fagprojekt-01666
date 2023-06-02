#%% Import packages and variables
import sys
sys.path.append('../')

from model.equation_interpreter import Equation
import numpy as np
import linecache

def save_data(file_names):
#Returns the abortion_rate, the length of each line in each sample and sample size_factors
#Saves them as matlab files.
    token_lengths = []
    abortion_rate = []
    sample_size = []
    sum_degree = []

    # Reads a file of answers and returns abortion rate and length of each expression
    for a,b in file_names:
        input_file =  f"../data_generation/7_sec_data/data/answers-{a}-{b}.txt"
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
        sum_degree.append(a+b)

    token_lengths = np.asarray(token_lengths)
    abortion_rate = np.asarray(abortion_rate)
    sample_size = np.asarray(sample_size)
    return token_lengths,abortion_rate,sample_size
