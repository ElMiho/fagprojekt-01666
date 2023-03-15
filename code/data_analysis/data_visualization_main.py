#%% Load data
from data_functions import *
import matplotlib.pyplot as plt
import pickle
from scipy.io import savemat

#Import file names and save data as pkl files.
with open('FileNames.pkl','rb') as f:
    file_names = pickle.load(f)
f.close()
token_lengths,abortion_rate,sample_size = save_data(file_names)

#Export data as matlab file
savemat('abort_rate.mat',{'abortion_rate': abortion_rate})
savemat('Token_Length.mat',{'Token_len': token_lengths})
savemat('Sample_size.mat',{'sample_size': sample_size})

#Save sample_size as pickle file
with open("SampleSize.pkl", "wb") as f:
    pickle.dump(sample_size, f)
f.close()

#%% Plot of histograms
#General distribution
fig, ax = plt.subplots()
for i in range(len(token_lengths)):
    ax.hist(token_lengths[i][token_lengths[i] != 0])
fig.suptitle('General distribution answer length 7 sec')
ax.set_xlabel('Length of answer')
ax.set_ylabel('Frequency')
plt.show()

#Plot general distribution
token_plot = token_lengths.flatten()
fig, ax = plt.subplots()
ax.hist(token_plot[token_plot != 0])
fig.suptitle('Single general distribution 7 sec')
ax.set_xlabel('Length of answer')
ax.set_ylabel('Frequency')
plt.show()
#%% Sample distributions
