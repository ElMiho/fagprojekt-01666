#%%
import pickle
import numpy as np
import math

#Load sample factors
with open('SampleSize.pkl','rb') as f:
    sample_size = pickle.load(f)
f.close()
np.savetxt('SampleSize.txt', sample_size, fmt='%i,%i,%.3f', delimiter=' ')
#%%
#Total number of samples
N = 40*10**6

#Sample from each category
n = N/len(sample_size)

sample_out = []
#Number of tokens in input library
nt = 45
#Percent of sample space in each category
pct_sample_space = []
strat_samples = []
for i in range(44):
    num = sample_size[i,0]
    den = sample_size[i,1]
    omega = nt**(num+den)
    size = sample_size[i,2]*n
    pct_sampled = (sample_size[i,2]*n)/omega

    if pct_sampled > 1:
        pct_sampled = 1
    elif pct_sampled < 10**(-2):
        pct_sampled = 0
    else: 
        strat_samples.append((num,den,omega,size))
    pct_sample_space.append((num,den,math.log(omega),math.log(size),pct_sampled))

#[Denominator, numerator, log[sample_space],log[sample_size], %sampled]
pct_sample_space = np.asarray(pct_sample_space)
np.set_printoptions(suppress=True, precision=2)
print(f"[Denominator, numerator, sample_space,sample_size, %sampled]\n\n{pct_sample_space}")
print(f"\n\n Examples where stratified sampling is neccesary\n{strat_samples}")
#%% Layout for stratified sampling