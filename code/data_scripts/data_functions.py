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

    token_lengths = np.asarray(token_lengths)
    abortion_rate = np.asarray(abortion_rate)
    sample_size = np.asarray(sample_size)
    return token_lengths,abortion_rate,sample_size



#%% Visualize abortion_rate and plot histograms
"""
fig, ax = plt.subplots()
for i in range(len(Token_Length)):
    ax.hist(Token_Length[i][Token_Length[i] != 0])
plt.show()


#%% Histograms of each test data
fig, axs = plt.subplots(5, 7)
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.hist(Token_Length[i][Token_Length[i] != 0])
    ax.set_title(file_names[i])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
fig.suptitle("Length vs [deg(p_n),deg(q_n)] ")
fig.tight_layout()
plt.show()

#%% Histograms of each test data
data_plot = []
fig, axs = plt.subplots(5, 7)
axs = axs.flatten()
for i, ax in enumerate(axs):
    data_plot.append(Data_Raw[i])
    data_plot[i] = Data_Raw[i][Data_Raw[i] != 0]
    data_plot[i] = Data_Raw[i][Data_Raw[i]<=50]
    ax.hist(data_plot[i])
    ax.set_title(file_names[i])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
fig.suptitle("Length vs [deg(p_n),deg(q_n)] ")
fig.tight_layout()
plt.show()


# %%
from scipy import stats
KS_test = []
Data_Raw = np.asarray(Data_Raw)
for i in range(len(Data_Raw)):
    test_data = Data_Raw[i][Data_Raw[i] != 0]
    min_x = min(test_data)
    max_x = max(test_data)
    KS_test.append(stats.kstest(test_data,'uniform',args=(min_x,max_x)))
# %%
"""
