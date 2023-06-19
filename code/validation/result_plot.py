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
with open(f"validation/save_data/saved_variables_1_5.pkl", 'rb') as f:
    loaded_variables8 = pickle.load(f)
# Access the loaded variables
found_distance8 = loaded_variables8['found_distance']
#total_counter8 = loaded_variables8['total_counter']
#non_valid_counter8 = loaded_variables8['non_valid_counter']
#degree_vector8 = loaded_variables8['degree_vector']

with open(f"validation/save_data/saved_variables_2_4.pkl", 'rb') as f:
    loaded_variables9 = pickle.load(f)
# Access the loaded variables
found_distance9 = loaded_variables9['found_distance']


#%%
zi = zip(found_distance1, found_distance2, found_distance3, found_distance4, found_distance5, found_distance6, found_distance7, found_distance8, found_distance9)
data = [row1 + row2 + row3 + row4+ row5+ row6+ row7 + row8 + row9 for row1, row2, row3, row4, row5, row6, row7, row8, row9 in zi]


#%%
# Create scatter plot with circular markers and no color
import random
idx = 0
labels = []
p = [0, 1, 2,3,4,5,6,7,8]
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
    [non_empt]*len(data[idx])
    X = [non_empt + (random.randint(0, 2)-1)/10 for _ in data[idx]]
    plt.scatter(X, data[idx], marker='o', label=label)
    idx += 1
    non_empt +=1
    hist_data.append(len(d))
    histdata.append(len(d))

   
# Create box plot
plt.boxplot(data_sort, positions=p, labels=labels)
plt.ylabel('Distance')
plt.xlabel('Degree of P(n) / Degree of Q(n)')
plt.title('Distance between correct and predicted evaluations - GPT2')
plt.xticks(rotation=-45)
plt.show()


#%%
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],histdata[i])
        
plt.bar(labels, hist_data, color ='blue',width = 0.4)
addlabels(labels, hist_data)
plt.title('Percentage of valid notation - GPT2')
plt.ylabel('Percentage valid Postfix Notation')
plt.xlabel('Degree of P(n) / Degree of Q(n)')
plt.show()

#%%
counts = [sum(1 for value in row if value <= 5) for row in data_sort]
            
            
def addlabels(x,y,data):
    for i in range(len(x)):
        plt.text(i,y[i],data[i])
# Width of each bar
bar_width = 0.35

# Positions of the left bar-boundaries
bar_positions = np.arange(len(labels))

# Create the figure and axis objects
fig, ax = plt.subplots()

# Plot the first bar plot
ax.bar(bar_positions, hist_data, width=bar_width, label='Valid postfix notation')

# Plot the second bar plot next to the first one
ax.bar(bar_positions + bar_width, counts, width=bar_width, label='Average distance less than 6')

addlabels(labels, hist_data,hist_data)
addlabels(labels, counts,counts)

# Customize the plot
ax.set_xticks(bar_positions + bar_width / 2)
ax.set_xticklabels(labels)
ax.legend()
plt.title('Percentage of valid notation - GPT2')
plt.ylabel('Percentage valid Postfix Notation')
plt.xlabel('Degree of P(n) / Degree of Q(n)')
# Display the plot
plt.show()


