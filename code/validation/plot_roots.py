import os
import pickle
import numpy as np 
import matplotlib.pyplot as plt



from validation.validation_medthods import posible_degrees
folder_path = "validation/save_data"  # Replace with the actual folder path containing the .pkl files

# Get a list of all .pkl files in the folder
file_list = [file for file in os.listdir(folder_path) if file.startswith("root") and file.endswith(".pkl")]

# Initialize a list to store the dictionaries
dis = []

# Concatenate the dictionaries from each file
for file in file_list:
    file_path = os.path.join(folder_path, file)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        dis.append(data['found_distance'])


zi = zip(dis[0], dis[1], dis[2], dis[3], dis[4])
data = [row1 + row2 + row3 + row4+ row5 for row1, row2, row3, row4, row5 in zi]

degree_vector = posible_degrees(10)

idx = 0
labels = []
p = [0, 1, 2,3,4]
data_sort = []
non_empt = 0
hist_data = []
histdata = []

for d in data:
    if len(d) == 0:
        idx += 1
        continue
    
    data_sort.append(d)
    label = f"{degree_vector[idx][0]}/{degree_vector[idx][1]}"
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

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],histdata[i])
        
plt.bar(labels, hist_data, color ='blue',width = 0.4)
addlabels(labels, hist_data)
plt.title('Percentage of valid notation \n in each category')
plt.ylabel('Percentage valid Postfix Notation')
plt.xlabel('Degree of P(n) / Degree of Q(n)')
plt.show()

