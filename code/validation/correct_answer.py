import sys
import networkx as nx
import matplotlib.pyplot as plt
import random

# SET PATH FOR OSX USERS
if sys.platform == 'darwin':
    sys.path.append('../code')

import model.tokenize_input as tokenize_input
import model.equation_interpreter as equation_interpreter

# Code from stackoverflow
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

test_string = "Pi^2/6"
eq = equation_interpreter.Equation.makeEquationFromString(test_string)
print(eq.notation)
eq.convertToPostfix()
print(eq.notation)

print(eq.tokenized_equation)
print(len(eq.tokenized_equation))

G = nx.Graph()
# print(eq.tokenized_equation[-5])

binary_operators = [
    "TT_PLUS",
    "TT_MINUS",
    "TT_MULTIPLY",
    "TT_DIVIDE", 
    "TT_POW"
]

idx = -1
while idx >= -len(eq.tokenized_equation):
    type = eq.tokenized_equation[idx].t_type
    value = eq.tokenized_equation[idx].t_value
    print(-idx, type, value)

    G.add_node(-idx, symbol=type)

    if type in binary_operators:
        G.add_edge(-idx, -(idx + 1))
        G.add_edge(-idx, (-idx + 2))

    idx -= 1
    
print(G.nodes.data())

plt.figure(1)
pos = nx.spring_layout(G)
# pos = hierarchy_pos(G, 1)
labels = nx.get_node_attributes(G, 'symbol')
nx.draw(G, labels=labels)
plt.show()

