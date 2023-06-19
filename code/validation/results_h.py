import sys
if False: #sys.platform == 'darwin':
    sys.path.append('../code')

from rnn import RNN
from validation.validation_medthods import *
from validation.validation_medthods import input_roots_num_den
from validation.validation_medthods import neural_network_validation
from validation.validation_medthods import roots_to_strings
from validation.validation_medthods import valid_equation
from validation.validation_medthods import posible_degrees

from model.equation_interpreter import Equation
from model.tokenize_input import input_string_to_tokenize_expression

from data_analysis.int_data.generate_plot import parse_line

from validation.TED import graph_from_postfix, TreeEditDistance

import pickle
from datetime import datetime

from tqdm import tqdm
import linecache
import argparse
from tqdm import tqdm
import os

#%%
# Parse command line arguments
hm = set()

# Write txt files with same name as folder - the monolith file
for datapoint in tqdm(open("data/THEFILE.txt")):
    hm.add(tuple(input_string_to_tokenize_expression(datapoint[:-1])))

hashmap = f"validation/vali_files/hashmap"
saved_variables = {'hashmap': hm}
with open(hashmap, 'wb+') as fi:
    pickle.dump(saved_variables, fi)
#%%
with open("validation/vali_files/hashmap", 'rb') as fes:
    loaded = pickle.load(fes)
    hashmap = loaded['hashmap']
#%%
#datastore
degree_vector = posible_degrees(10)
non_valid_counter = [0 for _ in degree_vector]
total_counter = [0 for _ in degree_vector]
found_distance = [[] for _ in degree_vector]


#To validate please add a string to the txt file it need to have same structure as megafile
deg1, deg2 = 0,2

testfile = f"validation/vali_files/answers-{deg1}-{deg2}-partition-1"


#open the test file
with open(testfile, "r") as file:
    lines = file.readlines()
    

#%%
current_time = datetime.now().time()
time_str = current_time.strftime("%H_%M")


for deg1 in range(0, 3):
    for deg2 in range(deg1+2, 7-deg1):
        degree_vector = posible_degrees(10)
        non_valid_counter = [0 for _ in degree_vector]
        total_counter = [0 for _ in degree_vector]
        found_distance = [[] for _ in degree_vector]
        
        if deg1 == 0 and deg2 == 2:
            continue
        
        testfile = f"validation/vali_files/answers-{deg1}-{deg2}-partition-1"
        filename = f"validation/save_data/TEDS_{deg1}_{deg2}.pkl"
        print(filename)
        
        
        vector_idx = degree_vector.index([deg1, deg2])
        
        loops = 0
        for line in lines:
            loops += 1
            if loops % 10 == 0:
                print(f"{loops} completed -- status : {len(found_distance[vector_idx])}")
                saved_variables = {'found_distance': found_distance}
                with open(filename, 'wb+') as f:
                    pickle.dump(saved_variables, f)
        
                    
            #reads the content from the line
            _, answer, _, roots = parse_line(line)
            roots_t = tuple(roots)
            
            
            if answer == "$Aborted":
                continue
            
            for h in hm:
                if h == roots_t:
                    print("Skipper")
                    continue
                
            # make the roots strings
            roots = roots_to_strings(roots)
            
                    
            #mesure the poly degrees
            numinator_degree, denorminator_degree = input_roots_num_den(roots)
            
            total_counter[vector_idx] += 1 
            
            # calls the neural network
            output_from_neural_network = neural_network_validation(roots)
            output_from_neural_network_RNN = RNN(roots)
            GPT = True
            RNN = True
            
            
            
            #test if the output is valid
            if not valid_equation(output_from_neural_network):
                GPT = False
            
            if not valid_equation(output_from_neural_network_RNN):
                RNN = False
            
            
            
            if GPT:
                #The sum file i can use have illigal anwsers and they behave in a way i cant predict
                try:
                    #genearte a tree from the anwser
                    answer = Equation.makeEquationFromString(answer)
                    answer.convertToPostfix()
                    tokens = answer.tokenized_equation
                    #if anwser is non legal
                    if len(tokens) == 0:
                        #Kunne være sjov at gemme line her
                        GPT = False
                    
                    if GPT:    
                        _, correct_tree = graph_from_postfix(tokens)
                        _, predicted_tree = graph_from_postfix(output_from_neural_network)
                        
                        #calculate distance
                        distance = TreeEditDistance().calculate(predicted_tree, correct_tree)
                        found_distance[vector_idx].append(distance[0])
                    
                except Exception:
                    None
            
            if RNN:
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
                    _, predicted_tree = graph_from_postfix(output_from_neural_network_RNN)
                    
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
import pickle


with open(f"validation/save_data/saved_variables_0_2.pkl", 'rb') as f:

    loaded_variables = pickle.load(f)

# Access the loaded variables
found_distance = loaded_variables['found_distance']
total_counter = loaded_variables['total_counter']
non_valid_counter = loaded_variables['non_valid_counter']
degree_vector = loaded_variables['degree_vector']


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

   

