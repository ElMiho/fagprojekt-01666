# Imports
import linecache
import time

from model.equation_interpreter import Equation
from model.vocabulary import vocabulary_answers, vocabulary_expressions
from model.tokenize_input import token_input_space, input_string_to_tokenize_expression

import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-input_file_answers", type=str, default="./data/answers-1000.txt")
parser.add_argument("-input_file_expressions", type=str, default="./data/expressions-1000.txt")
args = parser.parse_args()

# Paths to data files
input_file_answers = args.input_file_answers
input_file_expressions = args.input_file_expressions

# Prepend 'cleaned_' to output files
cleaned_file_answers = "/".join(input_file_answers.split("/")[:-1]) + "/cleaned_" + input_file_answers.split("/")[-1]
cleaned_file_expressions = "/".join(input_file_expressions.split("/")[:-1]) + "/cleaned_" + input_file_expressions.split("/")[-1]

# Run through every equation in the input files and delete rows with unknown tokens
dataset_size = sum(1 for _ in open(input_file_answers, 'rb'))

# Open and populate cleaned files and
f_cleaned_answers = open(cleaned_file_answers, "w")
f_cleaned_expressions = open(cleaned_file_expressions, "w")

start_time = time.time()
n_cleaned = 0
for line_number in range(1,dataset_size+1):
    # Get corresponding equation and expression
    raw_equation = linecache.getline(input_file_answers, line_number)
    raw_expression = linecache.getline(input_file_expressions, line_number)

    # Skip line if aborted error
    if raw_equation == "$Aborted\n": continue

    # Construct equation and convert to postfix
    equation = Equation.makeEquationFromString(raw_equation)
    if not equation.tokenized_equation: continue
    equation.convertToPostfix()
    if equation.notation == "infix": continue

    # Vectorize corresponding answer and expression
    vectorized_answers = vocabulary_answers.vectorize([token.t_type for token in equation.tokenized_equation])
    vectorized_expressions = vocabulary_expressions.vectorize([str(token) for token in input_string_to_tokenize_expression(raw_expression)])

    # Write them to cleaned data file
    f_cleaned_answers.write(f"{vectorized_answers}\n")
    f_cleaned_expressions.write(f"{vectorized_expressions}\n")
    n_cleaned += 1

    if line_number % 10_000 == 0:
        print(f"[{line_number}/{dataset_size}] --- Time since start: {time.time() - start_time} --- Successes: {n_cleaned} --- Fails: {line_number - n_cleaned + 1}")

print(vocabulary_expressions.token2index)