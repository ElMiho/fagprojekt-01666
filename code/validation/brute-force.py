import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# SET PATH FOR OSX USERS
if sys.platform == 'darwin':
    sys.path.append('../code')

def get_value(equation):
    pass

def get_best_guess(equation, real_value, span):
    num_vars = equation.count('Z')

    indexes = [0 for i in s2]
    for i, v in enumerate(s2):
        if v == "Z": indexes[i] = 1
    print(indexes)
    
    for index, idx in enumerate(indexes):
      if idx == 1:
        
    pass

if __name__ == '__main__':
    s1 = "Pi^2/6"
    s2 = "Pi^Z/Z + Log[Z]"


    s1.count("Z")
    get_best_guess(s2, 10, range(1,6))

