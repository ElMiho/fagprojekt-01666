import numpy as np

# Generate 44x3 array with [Deg numerator, Deg denominator, Number of samples, Total sample space]

#SampleSize.txt, 44x3 array with: [deg num, deg den, sample_factor]
sample_size = np.loadtxt('SampleSize.txt', delimiter=',')
sample_parameters = []

number_of_tokens = 34
total_samples = 40*10**6
categories = 44
n = total_samples//categories
for i in range(44):
    num = sample_size[i,0]
    den = sample_size[i,1]
    sample_factor = sample_size[i,2]
    number_of_samples = n*sample_factor
    omega = number_of_tokens**(num+den)     #Change according to different bases
    sample_parameters.append((num,den,number_of_samples,omega))
sample_parameters = np.asarray(sample_parameters)
titles = "Deg numerator, Deg denominator, Number of samples, Total sample space"
np.savetxt('SampleParameters.txt',sample_parameters, fmt='%i', delimiter=',', header=titles)
