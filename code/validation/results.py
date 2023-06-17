
from validation.validation_medthods import *
from validation.validation_medthods import input_roots_num_den
from validation.validation_medthods import neural_network_validation
from validation.validation_medthods import roots_to_strings
from validation.validation_medthods import valid_equation
from validation.validation_medthods import posible_degrees

from model.equation_interpreter import Equation

from data_analysis.int_data.generate_plot import parse_line

from validation.TED import graph_from_postfix, TreeEditDistance

import pickel
from datetime import datetime

#datastore
degree_vector = posible_degrees(10)
non_valid_counter = [0 for _ in degree_vector]
total_counter = [0 for _ in degree_vector]
found_distance = [[] for _ in degree_vector]


#To validate please add a string to the txt file it need to have same structure as megafile
testfile = "data_analysis/new_rational_data/megafile2_txt"

#open the test file
with open(testfile, "r") as file:
    lines = file.readlines()
    
loops = 0
succes_math = 0
#%%
current_time = datetime.now().time()
time_str = current_time.strftime("%H:%M:%S")
filename = f"validation/save_data/saved_variables_{time_str}.pkl"

#%%

for line in lines:
    loops += 1
    if loops % 50 == 0:
        print(f"{loops} completed -- status : {succes_math}")
        saved_variables = {'found_distance': found_distance, 'total_counter': total_counter, 'non_valid_counter': non_valid_counter, 'degree_vector': degree_vector}
        with open(filename, 'wb') as f:
            pickle.dump(saved_variables, f)
            
            
    #reads the content from the line
    _, answer, _, roots = parse_line(line)
    
    if answer == "$Aborted":
        continue
    
    # make the roots strings
    roots = roots_to_strings(roots)
    
    #mesure the poly degrees
    numinator_degree, denorminator_degree = input_roots_num_den(roots)
    vector_idx = degree_vector.index([numinator_degree, denorminator_degree])
    total_counter[vector_idx] += 1 
    
    # calls the neural network
    output_from_neural_network = neural_network_validation(roots)
        
    #test if the output is valid
    if not valid_equation(output_from_neural_network):
        non_valid_counter[vector_idx] += 1
        continue
    
    succes_math += 1
    
    #The sum file i can use have illigal anwsers and they behave in a way i cant predict
    try:
        #genearte a tree from the anwser
        answer = Equation.makeEquationFromString(answer)
        answer.convertToPostfix()
        tokens = answer.tokenized_equation
        #if anwser is non legal
        if len(tokens) == 0:
            #Kunne være sjov at gemme line her
            continue
    
        _, correct_tree = graph_from_postfix(tokens)
        _, predicted_tree = graph_from_postfix(output_from_neural_network)
        
        #calculate distance
        distance = TreeEditDistance().calculate(predicted_tree, correct_tree)
        found_distance[vector_idx].append(distance[0])
        
    except Exception:
        continue
    
    
    
#%%
import matplotlib.pyplot as plt
import numpy as np



# HER skal vi bare vælge hvilken del af dataen vi ønsker at plotte!!! Anbefaler måsle at lave flere
data = found_distance[0:10]

#%%
# Create scatter plot with circular markers and no color

idx = 0
labels = []
p = []
for d in data:
    label = f"{degree_vector[idx][0]}/{degree_vector[idx][1]}"
    labels.append(label)
    p.append(idx)
    plt.scatter([idx]*len(data[idx]), data[idx], marker='o', label=label)
    idx += 1
# Create box plot
plt.boxplot(data, positions=p, labels=labels)
plt.ylabel('Distance')
plt.xlabel('Degree x / Degree y')
plt.title('Boxplot of distance between correct and predicted evaluations')
plt.xticks(rotation=-45)
plt.show()

   

