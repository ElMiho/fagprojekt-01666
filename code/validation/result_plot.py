# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
import numpy as np





import pickle

#%%

with open(f"validation/save_data/saved_variables_0_2.pkl", 'rb') as f:
    loaded_variables1 = pickle.load(f)
# Access the loaded variables
found_distance1 = loaded_variables1['found_distance']
total_counter1 = loaded_variables1['total_counter']
non_valid_counter1 = loaded_variables1['non_valid_counter']
degree_vector1 = loaded_variables1['degree_vector']
    
with open(f"validation/save_data/saved_variables_0_3.pkl", 'rb') as f:
    loaded_variables2 = pickle.load(f)
# Access the loaded variables
found_distance2 = loaded_variables2['found_distance']
total_counter2 = loaded_variables2['total_counter']
non_valid_counter2 = loaded_variables2['non_valid_counter']
degree_vector2 = loaded_variables2['degree_vector']    

with open(f"validation/save_data/saved_variables_0_4.pkl", 'rb') as f:
    loaded_variables3 = pickle.load(f)
# Access the loaded variables
found_distance3 = loaded_variables3['found_distance']
total_counter3 = loaded_variables3['total_counter']
non_valid_counter3 = loaded_variables3['non_valid_counter']
degree_vector3 = loaded_variables3['degree_vector']

with open(f"validation/save_data/saved_variables_0_5.pkl", 'rb') as f:
    loaded_variables4 = pickle.load(f)
# Access the loaded variables
found_distance4 = loaded_variables4['found_distance']
total_counter4 = loaded_variables4['total_counter']
non_valid_counter4 = loaded_variables4['non_valid_counter']
degree_vector4 = loaded_variables4['degree_vector']    

with open(f"validation/save_data/saved_variables_0_6.pkl", 'rb') as f:
    loaded_variables5 = pickle.load(f)
# Access the loaded variables
found_distance5 = loaded_variables5['found_distance']
total_counter5 = loaded_variables5['total_counter']
non_valid_counter5 = loaded_variables5['non_valid_counter']
degree_vector5 = loaded_variables5['degree_vector']    

with open(f"validation/save_data/saved_variables_1_3.pkl", 'rb') as f:
    loaded_variables6 = pickle.load(f)
# Access the loaded variables
found_distance6 = loaded_variables6['found_distance']
total_counter6 = loaded_variables6['total_counter']
non_valid_counter6 = loaded_variables6['non_valid_counter']
degree_vector6 = loaded_variables6['degree_vector']    

with open(f"validation/save_data/saved_variables_1_4.pkl", 'rb') as f:
    loaded_variables7 = pickle.load(f)
# Access the loaded variables
found_distance7 = loaded_variables7['found_distance']
total_counter7 = loaded_variables7['total_counter']
non_valid_counter7 = loaded_variables7['non_valid_counter']
degree_vector7 = loaded_variables7['degree_vector']    

with open(f"validation/save_data/saved_variables_2_4.pkl", 'rb') as f:
    loaded_variables8 = pickle.load(f)
# Access the loaded variables
found_distance8 = loaded_variables8['found_distance']
total_counter8 = loaded_variables8['total_counter']
non_valid_counter8 = loaded_variables8['non_valid_counter']
degree_vector8 = loaded_variables8['degree_vector']

#%%
zi = zip(found_distance1, found_distance2, found_distance3, found_distance4, found_distance5, found_distance6, found_distance7, found_distance8)
data = [row1 + row2 + row3 + row4+ row5+ row6+ row7+ row8 for row1, row2, row3, row4, row5, row6, row7, row8 in zi]


#%%
# Create scatter plot with circular markers and no color

idx = 0
labels = []
p = [0, 1, 2,3,4,5,6,7]
data_sort = []
non_empt = 0
hist_data = []
histdata = []

for d in data:
    if len(d) == 0:
        idx += 1
        continue
    
    data_sort.append(d)
    label = f"{degree_vector1[idx][0]}/{degree_vector1[idx][1]}"
    labels.append(label)
    plt.scatter([non_empt]*len(data[idx]), data[idx], marker='o', label=label)
    idx += 1
    non_empt +=1
    hist_data.append(len(d)/100)
    histdata.append(len(d))

   
# Create box plot
plt.boxplot(data_sort, positions=p, labels=labels)
plt.ylabel('Distance')
plt.xlabel('Degree of P(n) / Degree of Q(n)')
plt.title('Distance between correct and predicted evaluations')
plt.xticks(rotation=-45)
plt.show()


#%%
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],histdata[i])
        
plt.bar(labels, hist_data, color ='blue',width = 0.4)
addlabels(labels, hist_data)
plt.title('Percentage of valid notation \n in each category')
plt.ylabel('Percentage valid Postfix Notation')
plt.xlabel('Degree of P(n) / Degree of Q(n)')
plt.show()

