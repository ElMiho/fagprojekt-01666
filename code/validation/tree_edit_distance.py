import numpy as np
import networkx as nx
import sys
import matplotlib.pyplot as plt
import collections

# Doesn't plot but rather saves the necessary commands
def plot_graph(G):
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'symbol')
    nx.draw(G, labels=labels)

class Node:
    def __init__(self, id = None, parent_node = None, value = None, children = []):
        self.parent_node = parent_node
        self.children = children # 0: left child, 1: right child
        self.value = value
        self.id = id
    
    # new_node is of the class Node
    def insert_child(self, new_node, before = False):
        if before: # Just because the other implementation does it
            self.children.insert(0, new_node)
        else:
            self.children.append(new_node)


class Tree:
    def __init__(self, nodes = [], root = None):
        self.nodes = nodes
        self.root = root
        
    def get_root(self):
        return self.nodes[-1]
    
    def get_ordering(self):
        pass

    def insert(self, parent_node: Node, new_node: Node):
        # Update children and parent
        parent_node.children.append(new_node)
        new_node.parent_node = parent_node
        # ... and add to the tree
        self.nodes.append(new_node)

    def l(self, i):
        current = self.root

    def to_nx_graph(self) -> nx.Graph:
        q = collections.deque()
        G = nx.Graph()
        
        root = self.get_root()
        G.add_node(root.id, symbol=root.value)
        for child in root.children:
            q.append(child)
        
        while q:
            current = q.pop()
            G.add_node(current.id, symbol=current.value)
            G.add_edge(current.parent_node.id, current.id)
            for child in current.children:
                q.append(child)

        return G

if __name__ == '__main__':
    n1 = Node(value="+")
    n2 = Node(value="a")
    n3 = Node(value="b")

    n1.children = [n2, n3]
    n3.parent_node = n1
    n2.parent_node = n1

    T = Tree([n2, n3, n1], root=n1)

    G = T.to_nx_graph()
    plot_graph(G)
    plt.show()