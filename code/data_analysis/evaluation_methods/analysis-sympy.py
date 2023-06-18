from sympy import I, oo, Sum, exp, pi
from sympy.abc import n
import sympy as sp
import itertools
import random
import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy

folder = "sympy-data"
files = os.listdir(folder)

data_matrix = np.zeros((len(files), 3))

for k, f in enumerate(files):
    current = open(f"{folder}/{f}", "r")
    content = current.read()
    decoded = json.loads(content)

    i, j = len(decoded[0][0]), len(decoded[0][1])
    
    total = len(decoded)
    counter = 0
    for [numerator_roots, denominator_roots, value] in decoded:
        if value == 'aborted':
            counter += 1

    abortion = counter / total

    data_matrix[k,:] = [i, j, abortion]

    # print(i, j, abortion)

print(data_matrix)

scipy.io.savemat('sympy.mat', {'sympy_data': data_matrix})