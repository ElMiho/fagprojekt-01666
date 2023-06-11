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
    def __init__(self, nodes = [], root = None, ordering = []):
        self.nodes = nodes
        self.root = root
        if ordering == []:
            self.ordering = self.get_ordering()
        else:
            self.ordering = ordering
    
    def get_ordering(self):
        if self.nodes == []:
            return []
        
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

    # l returns the leftmost decendant of i
    def l(self, i):
        ordering = self.ordering
        pos = ordering.index(i)
        current = self.nodes[pos]
        while current.children != []:
            current = current.children[0]

        return current
    
    def depth(self, i):
        pos = self.ordering.index(i)
        current = self.nodes[pos]
        counter = 0
        while current.parent_node != None:
            current = current.parent_node
            counter += 1

        return counter
    
    # anc returns all nodes that are on the parent path to the root node from
    # the i'th node
    def anc(self, i):
        pos = self.ordering.index(i)
        
        current = self.nodes[pos]
        depth = self.depth(i)
        
        anc_list = [current]
        for i in range(depth + 1):
            current = current.parent_node
            if current != None:
                anc_list.append(current)

        return anc_list
    
    def subforest(self, i, j):
        """
        corresponding to T[i...j]
        """
        if i > j: return Tree() # Find out if this results in problems!

        subordering = []
        # list_of_indexes = []
        sub_node_list = []
        for idx, value in enumerate(self.ordering):
            if value >= i and value <= j:
                # list_of_indexes.append(idx)
                node = self.nodes[idx]
                sub_node_list.append(node)
                subordering.append(self.ordering[idx])

        subtree = Tree(nodes=sub_node_list, ordering=subordering)

        return subtree

    def forest(self, i):
        return self.subforest(1, i)
    
    def tree(self, i):
        return self.subforest(self.l(i), i)
    
    def LR_keyroots(self):
        keyroots = []
        
        ordering = self.ordering
        keyroots.append(max(ordering))

        for k in range(1, len(self.nodes)): # excluding the last element since this is the root
            k_node = self.nodes[
                ordering.index(k)
            ]
            parent_k_node = k_node.parent_node
            parent_k = ordering[
                self.nodes.index(parent_k_node)
            ]

            if self.l(k) != self.l(parent_k):
                keyroots.append(k)

        return sorted(keyroots)

    def to_nx_di_graph(self) -> nx.DiGraph:
        q = collections.deque()
        G = nx.DiGraph()

        ordering = self.ordering
        for idx, node in enumerate(self.nodes):
            node.id = ordering[idx]
        
        for node in self.nodes:
            G.add_node(node.id, symbol=f"{node.value}({node.id})")

        for node in self.nodes:
            for child in node.children:
                if child in self.nodes:
                    G.add_edge(node.id, child.id)

            if node.parent_node in self.nodes:
                G.add_edge(node.parent_node.id, node.id)

        return G
    
def tree_edit_distance(T1: Tree, T2: Tree):
    cost = 1
    LR_T1 = T1.LR_keyroots()
    LR_T2 = T2.LR_keyroots()

    permanent_forestdist = np.matrix((
        len(T1.nodes), len(T2.nodes)
    ))

    permanent_forestdist[0, 0] = 0

    def forestdist(T1: Tree, i_1: int, i_2: int, T2: Tree, j_1: int, j_2: int):
        if T1.nodes == [] and T2.nodes == []:
            return 0
        if T2.nodes == []:
            new_T1_subforest = T1.subforest(i_1, i_2 - 1)
            return forestdist(new_T1_subforest, i_1, i_2 - 1, T2, j_1, j_2) + cost
        if T1.nodes == []:
            new_T2_subforest = T2.subforest(j_1, j_2 - 1)
            return forestdist(T1, i_1, i_2, new_T2_subforest, j_1, j_2 - 1) + cost
        
        T1_possibility = T1.subforest(i_1, i_2 - 1)
        T2_possibility = T2.subforest(j_1, j_2 - 1)

        options = [
            forestdist(T1_possibility, i_1, i_2 - 1, T2, j_1, j_2) + cost, 
            forestdist(T1, i_1, i_2, T2_possibility, j_1, j_2 - 1) + cost, 
            forestdist(T1_possibility, i_1, i_2 - 1, T2_possibility, j_1, j_2 - 1) + cost
        ]

        return min(options)

    def treedist(T1: Tree, T2: Tree, i, j):
        ordering_T1 = T1.ordering
        ordering_T2 = T2.ordering
        
        l_i_node = T1.l(i)
        l_i = ordering_T1[T1.nodes.index(l_i_node)]
        for i_1 in range(l_i, i + 1):
            forestdist(T1.subforest(l_i, i_1), l_i, i_1, Tree(), 0, 0)

        l_j_node = T2.l(j)
        l_j = ordering_T2[T2.nodes.index(l_j_node)]
        for j_1 in range(l_j, j + 1):
            forestdist(Tree(), 0, 0, T2.subforest(l_j, j_1), l_j, j_1)


    for i in LR_T1:
        for j in LR_T2:
            treedist(T1, T2, i, j)

# def forestdist(T1: Tree, i_1: int, i_2: int, T2: Tree, j_1: int, j_2: int):
#     cost = 1
#     if T1.nodes == [] and T2.nodes == []:
#         return 0
#     if T2.nodes == []:
#         new_T1_subforest = T1.subforest(i_1, i_2 - 1)
#         return forestdist(new_T1_subforest, i_1, i_2 - 1, T2, j_1, j_2) + cost
#     if T1.nodes == []:
#         new_T2_subforest = T2.subforest(j_1, j_2 - 1)
#         return forestdist(T1, i_1, i_2, new_T2_subforest, j_1, j_2 - 1) + cost
    
#     T1_possibility = T1.subforest(i_1, i_2 - 1)
#     T2_possibility = T2.subforest(j_1, j_2 - 1)

#     options = [
#         forestdist(T1_possibility, i_1, i_2 - 1, T2, j_1, j_2) + cost, 
#         forestdist(T1, i_1, i_2, T2_possibility, j_1, j_2 - 1) + cost, 
#         forestdist(T1_possibility, i_1, i_2 - 1, T2_possibility, j_1, j_2 - 1) + cost
#     ]

#     return min(options)


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

    T1 = Tree([b, a, c, d, e, f], root = f)

    print(T1.LR_keyroots())

    for node in T1.anc(4):
        print(node.value)

    print(T1.ordering)

    print("subtree stuff 1..3")
    sub = T1.subforest(1, 3)
    # print(sub.nodes)
    for n in sub.nodes:
        print(n.value)
    print(sub.ordering)

    sub2 = sub.subforest(1, 2)

    plt.figure(1)
    J = T1.to_nx_di_graph()
    plot_graph(J)

    plt.figure(2)
    U = sub.to_nx_di_graph()
    plot_graph(U)

    plt.figure(3)
    L = sub2.to_nx_di_graph()
    plot_graph(L)
    
    plt.show()