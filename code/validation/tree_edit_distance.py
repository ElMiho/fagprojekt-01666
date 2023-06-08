import numpy as np
import networkx as nx
# from networkx.drawing.nx_agraph import graphviz_layout
import sys
import matplotlib.pyplot as plt
import collections

# Doesn't plot but rather saves the necessary commands
def plot_graph(G):
    pos = nx.spring_layout(G)
    # pos = graphviz_layout(G, prog='dot')
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
    
    def get_ordering(self):  
        idx_list = [0] * len(self.nodes)
        idx = 1

        def has_children_been_visited(node: Node):
            all_visited = True
            for child in node.children:
                pos = self.nodes.index(child)
                if idx_list[pos] == 0: 
                    all_visited = False
                    break

            if all_visited:
                return None
            else:
                return child

        # find fist node
        current = self.root
        while current.children != []:
            current = current.children[0]
        
        # and visit the nodes in left-to-right postorder numbering
        while 0 in idx_list:
            if has_children_been_visited(current) == None:
                # visit the current
                pos = self.nodes.index(current)
                idx_list[pos] = idx
                idx +=1 
                
                # and go to parent
                current = current.parent_node
            else:
                current = has_children_been_visited(current)

        return idx_list

    def insert(self, parent_node: Node, new_node: Node):
        # Update children and parent
        parent_node.children.append(new_node)
        new_node.parent_node = parent_node
        # ... and add to the tree
        self.nodes.append(new_node)

    def l(self, i):
        ordering = self.get_ordering()
        pos = ordering.index(i)
        current = self.nodes[pos]
        while current.children != []:
            current = current.children[0]

        return current
    
    def depth(self, i):
        ordering = self.get_ordering()
        pos = ordering.index(i)
        current = self.nodes[pos]
        counter = 0
        while current.parent_node != None:
            current = current.parent_node
            counter += 1

        return counter
    
    def anc(self, i):
        anc_list = []
        ordering = self.get_ordering()
        pos = ordering.index(i)
        
        current = self.nodes[pos]
        depth = self.depth(i)
        
        anc_list.append(current)
        for i in range(0, depth + 1):
            current = current.parent_node
            if current != None:
                anc_list.append(current)

        return anc_list
    
    def subforest(self, i, j):
        """
        corresponding to T[i...j]
        """
        if i > j: return [] # Find out if this results in problems!
        
        ordering = self.get_ordering()
        list_of_indexes = []
        for idx, value in enumerate(ordering):
            if value >= i and value <= j:
                list_of_indexes.append(idx)

        return [self.nodes[i] for i in list_of_indexes]

    def forest(self, i):
        return self.subforest(1, i)
    
    def tree(self, i):
        return self.subforest(self.l(i), i)
    
    def LR_keyroots(self):
        keyroots = []
        
        ordering = self.get_ordering()
        keyroots.append(max(ordering))

        for k in range(1, len(self.nodes) + 1):
            k_node = self.nodes[
                ordering.index(k)
            ]
            if k_node != self.root:
                parent_k_node = k_node.parent_node
                parent_k = ordering[
                    self.nodes.index(parent_k_node)
                ]

                if self.l(k) != self.l(parent_k):
                    keyroots.append(k)

        return keyroots

    def to_nx_di_graph(self) -> nx.DiGraph:
        q = collections.deque()
        G = nx.DiGraph()

        ordering = self.get_ordering()
        for idx, node in enumerate(self.nodes):
            node.id = ordering[idx]
        
        root = self.root
        G.add_node(root.id, symbol=f"{root.value}({root.id})")
        for child in root.children:
            q.append(child)

        while q:
            current = q.pop()
            G.add_node(current.id, symbol=f"{current.value}({current.id})")
            G.add_edge(current.parent_node.id, current.id)
            for child in current.children:
                q.append(child)

        return G
    
def treedist(T1: Tree, T2: Tree, i: int, j: int, treedist_matrix: np.matrix):
    """
    i in T1 and j in T2
    """
    for i_1_node in T1.anc(i):
        "some check"
    for j_1_node in T2.anc(j):
        "some check"

    for i_1_node in T1.anc(i):
        "some check"
        for j_1_node in T2.anc(j):
            ordering_T1 = T1.get_ordering()
            ordering_T2 = T2.get_ordering()

            pos_i_1 = T1.nodes.index(i_1_node)
            i_1 = ordering_T1[pos_i_1]

            pos_j_1 = T2.nodes.index(j_1_node)
            j_1 = ordering_T2[pos_j_1]

            if T1.l(i_1) == T1.l(i) and T2.l(j_1) == T2.l(j):
                "lots of calculations"
            else:
                "lots of calculations"

def forestdist(T1: Tree, T2: Tree):
    if T1.nodes == [] and T2.nodes == []:
        return 0

# poor mans test cases and playing around
if __name__ == '__main__':
    # expression: cos(c - d) + b
    n1 = Node(value="+")
    n2 = Node(value="cos")
    n3 = Node(value="-")
    n4 = Node(value="c")
    n5 = Node(value="d")
    n6 = Node(value="b")

    n1.children = [n2, n6]
    n2.parent_node = n1
    n6.parent_node = n1

    n2.children = [n3]
    n3.parent_node = n2

    n3.children = [n4, n5]
    n4.parent_node = n3
    n5.parent_node = n3

    T = Tree([n1, n2, n3, n4, n5, n6], root = n1)

    # G = T.to_nx_di_graph()
    # plot_graph(G)

    # print("test ordering")
    # print(T.get_ordering())

    # print("test l(5)")
    # print(
    #     T.l(5).value
    # )

    # print("test depth")
    # print(
    #     [(3, T.depth(3)), (4, T.depth(4)), 
    #     (5, T.depth(5)), (1, T.depth(1))]
    # )

    # print("test anc")
    # ordering = T.get_ordering()
    # anc = T.anc(1)
    # for n in anc:
    #     pos = T.nodes.index(n)
    #     print(ordering[pos], n.value)

    # print("test subforest")
    # res = T.subforest(3, 5)
    # for n in res:
    #     pos = T.nodes.index(n)
    #     print(ordering[pos], n.value)

    # Fig 4 node example
    f = Node(value="f")
    d = Node(value="d")
    e = Node(value="e")
    a = Node(value="a")
    c = Node(value="c")
    b = Node(value="b")

    f.children = [d, e]
    d.parent_node = f
    e.parent_node = f

    d.children = [a, c]
    a.parent_node = d
    c.parent_node = d

    c.children = [b]
    b.parent_node = c

    T1 = Tree([a, b, c, d, e, f], root = f)

    print(T1.LR_keyroots())

    J = T1.to_nx_di_graph()
    plot_graph(J)
    plt.show()