# Imports
from typing import List
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import uuid
import copy

from model.tokens import *

unary_operators = [
    "TT_SQRT",
    "TT_SIN",
    "TT_COS",
    "TT_TAN", 
    "TT_LOG"
]

binary_operators = [
    "TT_PLUS",
    "TT_MINUS",
    "TT_MULTIPLY",
    "TT_DIVIDE", 
    "TT_POW"
]

commutative_operators = [
    "TT_PLUS",
    "TT_MULTIPLY"
]


# Tree Node Class
class Node:
    def __init__(self, val:str, left=None, right=None, parent=None):
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent
        self.ID = uuid.uuid4()
        
    def __eq__(self, other):
        return isinstance(other, Node) and self.ID == other.ID
        
    def __repr__(self):
        return f"[{self.val} | Left: {self.left} | Right: {self.right} | Parent: {self.parent.val if self.parent else None}]"
    
    # Does create new ID's - so not a complete deepcopy
    def __copy__(self, parent=None):
        root = Node(self.val, parent=parent)
        root.left = copy.copy(self.left) if self.left else None
        root.right = copy.copy(self.right) if self.right else None
        
        if root.right:
            root.right.parent = root
        if root.left:
            root.left.parent = root
        
        return root

# Input is token list - defaults to right side - returns the last position at which 
def graph_from_postfix(postfix:List[Token], parent=None) -> tuple:
    pointer = len(postfix)-1
    token = postfix[pointer].t_type
    root = Node(token)
    root.parent = parent
    if token in unary_operators:
        pointer, root.right = graph_from_postfix(postfix[:pointer], root)
    elif token in binary_operators:
        pointer, root.right = graph_from_postfix(postfix[:pointer], root)
        pointer, root.left = graph_from_postfix(postfix[:pointer], root)
    
    return pointer, root
 
# This is to get a fingerprint of each graph
def graph_to_postfix(root:Node):
    if not root: return []
    postfix = graph_to_postfix(root.left)
    postfix += graph_to_postfix(root.right)
    postfix += [root.val]
    return postfix

def node_count(root):
    if not root:
        return 0
    
    return node_count(root.left) + node_count(root.right) + 1


# Shows the graph
def show_graph(root:Node, root_x=0, root_y=0, level=1, color="orange"):
    plt.axis("off")
    dist_mid = np.sqrt(2)
    theta = 45/180 * np.pi / (0.6*level + 0.4) # I wanted do 1/level, but it was horrendous for big graphs
    font_space = 0.3
        
    # Plot node
    plt.text(root_x, root_y, f"${token2symbol[root.val]}$",
              bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},
              ha='center', va='center', 
             fontsize=18)
    
    # Plot children
    if root.left and root.right:
        x_left, y_left = root_x - dist_mid * np.tan(theta) + font_space, root_y - dist_mid - font_space
        x_right, y_right = root_x + dist_mid * np.tan(theta) - font_space, y_left
        
        # Show line and then recursively show left side
        plt.plot((root_x, x_left), (root_y, y_left), color=color)
        show_graph(root.left, x_left, y_left, level+1, color)
        
        # Show line and then recursively show right side
        plt.plot((root_x, x_right), (root_y, y_right), color=color)
        show_graph(root.right, x_right, y_right, level+1, color)
    elif root.right:
        x_mid, y_mid = root_x, root_y - dist_mid
        plt.plot((root_x, x_mid), (root_y, y_mid), color=color)
        # Recursively show graph
        show_graph(root.right, x_mid, y_mid, level+1, color)
        

# Assumes that 'parent' isn't a leaf node - note that the 'node' parameter only needs to be specified if parent is None
def node_insert(node, to_insert):
    parent = node.parent
    if not parent:
        to_insert.right = node
        node.parent = to_insert
        return
    
    if parent.right == node:
        parent.right = to_insert
        to_insert.right = node
        
    elif parent.left == node:
        parent.left = to_insert
        to_insert.right = node
        
        
# Replace one node value with another 
def node_replace(node, value_to):
    node.val = value_to
        
        
# Delete node and replace with right or left subnode (if applicable) - note that it is assummed not to be the root (with good reason)
def node_delete(node, replace_w_right=True):
    if replace_w_right:
        if node.parent.right == node:
            node.parent.right = node.right
        else:
            node.parent.left = node.right
    else:
        if node.parent.right == node:
            node.parent.right = node.left
        else:
            node.parent.left = node.left
            
    # This is just a nicety for later
    node.parent = None
        
        
# Delete whole subtree
def node_delete_all(node):
    # If is root
    if not node.parent: 
        node = None
        
    elif node.parent.right == node:
        node.parent.right = None
        
    elif node.parent.left == node:
        node.parent.left = None
    
    # This is just a nicety for later
    node.parent = None
        
        
# Swaps one child with another
def node_swap(node):
    node.left, node.right = node.right, node.left
        
        
# # Tree Edit Distance Algorithm
# `root_pred` is the root of the predicted tree

# `root_correct` is the root of the correct tree

# Does a simultaneous BFS and each time a discrepancy is encountered, every tool in the arsenal above is thrown at it (not actually, only the result - like for lehvenstein), and the action used is recorded so the result also provides what operations are needed where


# An important note is that because of the `Replace and Swap` operator TED(a,b) is not necessarily TED(b,a), based on the commutativity of the replacement operator
        
class TreeEditDistance:
    def __init__(self):
        self.cache = {}
        
    def calculate(self, root_predict, root_correct):
        key = (",".join(graph_to_postfix(root_predict)), ",".join(graph_to_postfix(root_correct)))
        if key in self.cache:
            return self.cache[key]
        
        # Base case
        if not root_predict or not root_correct:
            return (max(node_count(root_predict), node_count(root_correct)), "Da") # Delete all none-null nodes
        
        # Replace cost is 1 if the root nodes contain different values
        replace_cost = int(root_predict.val != root_correct.val)
        
        # Recurse and calculate the minimum distance
        min_dist, operation = min( 
                       (self.calculate(root_predict.left, root_correct.left)[0] + self.calculate(root_predict.right, root_correct.right)[0] + replace_cost, "R"), # Replace Node
                       (self.calculate(root_predict.left, root_correct.right)[0] + self.calculate(root_predict.right, root_correct.left)[0] + replace_cost + (1 if not root_correct.val in commutative_operators else 0), "RS"), # Replace Node + Swap Node
                       (self.calculate(None, root_correct.left)[0] + self.calculate(root_predict, root_correct.right)[0] + 1, "I"), # Insert Node
                       (self.calculate(root_predict.left, root_correct)[0] + node_count(root_predict.right) + 1, "Dl"), # Delete Node and replace with left
                       (self.calculate(root_predict.right, root_correct)[0] + node_count(root_predict.left) + 1, "Dr"), # Delete Node and replace with right
                       key = lambda v: v[0]
                      )

        self.cache[key] = (min_dist, operation)
        return (min_dist, operation)
    
    def __call__(self, root_predict, root_correct):
        return self.calculate(root_predict, root_correct)[0]
        
def get_steps(predict_graph, correct_graph):
    steps = [(predict_graph, None)]
    # Instantiate our calculator
    TED = TreeEditDistance()

    # Some meta-data
    total_dist = 0

    # Instantiate the subtree-root stack
    root_predict, root_correct = copy.copy(predict_graph), copy.copy(correct_graph)
    stack = [(root_predict, root_correct)] # Stores the node-pairs for subtrees

    # While there are still operations to perform
    idx = 1
    while stack:
        node_predict, node_correct = stack.pop()
        if not node_predict and not node_correct: continue

        # Lookup operation and distance in cache
        dist, operation = TED.calculate(node_predict, node_correct)
        total_dist += dist

        val_pred = node_predict.val if node_predict else None
        val_cor = node_correct.val if node_correct else None

        # Modify the graph according to 'operation'
        ## Replace node
        if operation == "R":
            flag = node_predict and node_correct and node_predict.val == node_correct.val
            node_replace(node_predict, node_correct.val)
            # Append to stack
            stack.append((node_predict.left if node_predict else None, node_correct.left if node_correct else None))
            stack.append((node_predict.right if node_predict else None, node_correct.right if node_correct else None))
            if flag: 
                continue

        ## Replace node and swap
        elif operation == "RS":
            node_swap(node_predict)
            node_replace(node_predict, node_correct.val)
            # Append swapped to stack
            stack.append((node_predict.left if node_predict else None, node_correct.left if node_correct else None))
            stack.append((node_predict.right if node_predict else None, node_correct.right if node_correct else None))

        ## Insert node
        elif operation == "I":
            node_insert(node_predict, Node(node_correct.val))
            # Append to stack
            stack.append((None, node_correct.left if node_correct else None))
            stack.append((node_predict, node_correct.right if node_correct else None))

        ## Delete and replace with left
        elif operation == "Dl":
            node_delete(node_predict, False)
            # Append to stack
            stack.append((node_predict.left if node_predict else None, node_correct))

        ## Delete and replace with right
        elif operation == "Dr":
            node_delete(node_predict)
            # Append to stack
            stack.append((node_predict.right if node_predict else None, node_correct))

        ## Delete all
        elif operation == "Da":
            # In case a previous "Da" has already deleted this subtree, skip
            root_before = node_predict
            while root_before and root_before.parent: 
                root_before = root_before.parent
                if root_before != root_predict: 
                    continue

            # Delete all
            node_delete_all(node_predict)

        # Save the modified graph
        steps.append((copy.copy(root_predict), (idx, token2symbol[val_pred], token2symbol[val_cor])))
    
    steps.append((correct_graph, None))
    return steps

        
        
        
        