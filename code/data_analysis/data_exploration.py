import linecache
from model.equation_interpreter import Equation
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


file_names = [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 
  10), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 
  5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 6), (4, 7), (4, 
  8), (4, 9), (4, 10), (5, 7), (5, 8), (5, 9), (5, 10), (6, 8), (6, 
  9), (6, 10), (7, 9), (7, 10), (8, 10)]

data_plot = []
for a,b in [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 
  10), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 5), (3, 6), (3,7), (3, 8), (3, 9), (3, 10),(4, 6), (4, 7), (4, 
  8), (4, 9)]:
    input_file = f"data_generation/data/answers-{a}-{b}.txt"
    number_of_lines = sum(1 for i in open(input_file, 'rb'))
    num_non_zeros = number_of_lines

    token_lengths = 0
    for line_index in range(1,number_of_lines+1):
        raw_equation = linecache.getline(input_file, line_index)
        equation = Equation.makeEquationFromString(raw_equation)
        tokenized_length = len(equation.tokenized_equation)
        print(f"Raw equation: {raw_equation}\n Tokenized equation: {equation.tokenized_equation}\n Length: {tokenized_length}")
        print("\n\n")
        if tokenized_length == 0:
            num_non_zeros -= 1
        token_lengths += tokenized_length

    data_plot.append((a, b, token_lengths / num_non_zeros))

data_plot = np.asarray(data_plot)