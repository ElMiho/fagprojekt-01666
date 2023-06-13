import numpy as np
import networkx as nx
# from networkx.drawing.nx_agraph import graphviz_layout
import sys
import matplotlib.pyplot as plt
import collections
import pydot


# Doesn't plot but rather saves the necessary commands
def plot_graph(G):
    pos = nx.spring_layout(G)
    # pos = graphviz_layout(G, prog='dot')
    labels = nx.get_node_attributes(G, 'symbol')
    nx.draw_networkx(G, labels=labels)

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
        
        # lexicographic ordering of nodes
        def is_node_commutative(node):
            return node.value in ["+", "*"]

        for node in self.nodes:
            if is_node_commutative(node):
                [child_1, child_2] = node.children
                if child_1.value < child_2.value:
                    node.children = [child_1, child_2]
                else:
                    node.children = [child_2, child_1]

        # ... and left-to-right postorder numbering
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
    
    def node_to_i(self, node):
        ordering = self.ordering
        pos = self.nodes.index(node)

        return ordering[pos]
    
    def i_to_node(self, i):
        pos = self.ordering.index(i)
        return self.nodes[pos]
    
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
        l_i = self.node_to_i(self.l(i))
        new_tree = self.subforest(l_i, i)
        root_id = new_tree.ordering.index(
            max(new_tree.ordering)
        )
        root = new_tree.nodes[root_id]
        new_tree.root = root
        new_tree.ordering = new_tree.get_ordering()

        return new_tree
    
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
    
    def node(self, i):
        return self.nodes[i-1]
    
def cost_function(n1: Node, n2: Node) -> int:
    if n1.value == n2.value:
        return 0
    else:
        return 1
    
def tree_edit_distance(T1: Tree, T2: Tree):
    LR_T1 = T1.LR_keyroots()
    LR_T2 = T2.LR_keyroots()

    T1_size = len(T1.nodes)
    T2_size = len(T2.nodes)

    treedist = -1*np.ones((T1_size + 1, T2_size + 1))

    # save the forestdist for each (i, j)
    forestdist_dict = dict()
    # saved as (i, j, choose operation)
    operations = []
    def compute_treedist(i, j):
        forestdist = -1*np.ones((i + 1, j + 1))
        l_i_node = T1.l(i)
        l_i = T1.node_to_i(l_i_node)
        l_j_node = T2.l(j)
        l_j = T2.node_to_i(l_j_node)

        forestdist[0, 0] = 0

        def fd_index(x, y):
            if x <= y:
                return y
            else:
                return 0

        EMPTY = Node()
        for i_1 in range(l_i, i + 1):
            forestdist[i_1, 0] = forestdist[fd_index(l_i, i_1 - 1), 0] + cost_function(T1.node(i_1), EMPTY)
        for j_1 in range(l_j, j + 1):
            forestdist[0, j_1] = forestdist[0, fd_index(l_j, j_1 - 1)] + cost_function(EMPTY, T2.node(j_1))

        for i_1 in range(l_i, i + 1):
            for j_1 in range(l_j, j + 1):
                if T1.l(i_1) == T1.l(i) and T2.l(j_1) == T2.l(j):
                    
                    option_1 = (1, forestdist[fd_index(l_i, i_1 - 1), fd_index(l_j, j_1)] + cost_function(T1.i_to_node(i_1), EMPTY))
                    option_2 = (2, forestdist[fd_index(l_i, i_1), fd_index(l_j, j_1 - 1)] + cost_function(EMPTY, T2.i_to_node(j_1)))
                    option_3 = (3, forestdist[fd_index(l_i, i_1 - 1), fd_index(l_j, j_1 - 1)] + cost_function(T1.i_to_node(i_1), T2.i_to_node(j_1)))

                    choosen_one = min([option_1, option_2, option_3], key = lambda t: t[1])

                    forestdist[i_1, j_1] = choosen_one[1]
                    treedist[i_1, j_1] = forestdist[i_1, j_1]
                    
                    # save operation
                    operations.append((i, j, choosen_one[0]))
                else:
                    option_1 = (1, forestdist[fd_index(l_i, i_1 - 1), fd_index(l_j, j_1)] + cost_function(T1.i_to_node(i_1), EMPTY))
                    option_2 = (2, forestdist[fd_index(l_i, i_1), fd_index(l_j, j_1 - 1)] + cost_function(EMPTY, T2.i_to_node(j_1)))
                    option_3 = (3, forestdist[fd_index(l_i, i_1 - 1), fd_index(l_j, j_1 - 1)] + treedist[i_1, j_1])

                    choosen_one = min([option_1, option_2, option_3], key = lambda t: t[1])
                    forestdist[i_1, j_1] = choosen_one[1]

                    # save operation
                    operations.append((i, j, choosen_one[0]))

        forestdist_dict[(i, j)] = forestdist

    for i in LR_T1:
        for j in LR_T2:
            compute_treedist(i, j)

    return treedist, operations, forestdist_dict

def tree_node_diff(T_original: Tree, T_new: Tree) -> list[Node]:
    nodes = []
    for n in T_original.nodes:
        if n not in T_new.nodes:
            nodes.append(n)

    return nodes

def construct_path(path_matrix: np.matrix, i0 = 0, j0 = 0) -> list[tuple]:
    """
    works by constructing the path backwards
    """
    m, n = path_matrix.shape
    path = []
    path.append((m-1, n-1))

    def next_element(path_matrix: np.matrix, i: int, j: int) -> None:
        if i == i0 and j == j0:
            return None
        else:
            options = [
                ((i, j - 1), path_matrix[i, j - 1]), 
                ((i - 1, j - 1), path_matrix[i - 1, j - 1]),
                ((i - 1, j), path_matrix[i - 1, j])
            ]
            # pick by the smallest value in the matrix
            next_position = min(options, key = lambda t: t[1])
            path.insert(0, next_position[0])
            next_element(path_matrix, next_position[0][0], next_position[0][1])
    
    next_element(path_matrix, m-1, n-1)
    return path

def path_to_operations(path_matrix: np.matrix, path: list[int]):
    pass

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

    # # Fig 4 node example
    # f = Node(value="f")
    # d = Node(value="d")
    # e = Node(value="e")
    # a = Node(value="a")
    # c = Node(value="c")
    # b = Node(value="b")

    # f.children = [d, e]
    # d.parent_node = f
    # e.parent_node = f

    # d.children = [a, c]
    # a.parent_node = d
    # c.parent_node = d

    # c.children = [b]
    # b.parent_node = c

    # T1 = Tree([e, d, f, a, b, c], root = f)
    # # T1_sub = T1.subforest(1, 4)
    # # T1_sub_2 = T1.subforest(2, 3)

    # plt.figure("T1")
    # T1_nx = T1.to_nx_di_graph()
    # plot_graph(T1_nx)

    # f2 = Node(value="f")
    # d2 = Node(value="d")
    # e2 = Node(value="e")
    # a2 = Node(value="a")
    # c2 = Node(value="c")
    # b2 = Node(value="b")
    
    # f2.children = [c2, e2]
    # c2.parent_node = f2
    # e2.parent_node = f2

    # c2.children = [d2]
    # d2.parent_node = c2

    # d2.children = [a2, b2]
    # a2.parent_node = d2
    # b2.parent_node = d2
    
    # T2 = Tree([f2, d2, e2, a2, c2, b2], root=f2)
    # treedist, operations, forestdist_dict = tree_edit_distance(T1, T2)
    # print(f"treedist:\n{treedist}")
    # path_treedist = construct_path(treedist, i0=1, j0=1)
    # print(f"path_treedist:\n{path_treedist}")

    # print("\n")

    # m, n = len(T1.nodes), len(T2.nodes)
    # print(f"forestdist for {m, n} (for path)")
    # print(forestdist_dict[(m, n)])

    # path_forestdist = construct_path(forestdist_dict[(m, n)], i0=0, j0=0)
    # print(f"path_forestdist:\n{path_forestdist}")

    # # plotting
    # plt.figure("T2")
    # T2_nx = T2.to_nx_di_graph()
    # plot_graph(T2_nx)

    # T1_sub = T1.tree(3)
    # plt.figure("T1.tree(3)")
    # T1_sub_nx = T1_sub.to_nx_di_graph()
    # plot_graph(T1_sub_nx)

    # plt.figure("T1[1..4]")
    # T1_sub_nx = T1_sub.to_nx_di_graph()
    # plot_graph(T1_sub_nx)

    # plt.figure("T1[2..3]")
    # T1_sub_2_nx = T1_sub_2.to_nx_di_graph()
    # plot_graph(T1_sub_2_nx)

    # nodes = tree_node_diff(T1, T1_sub)
    # [print(n.value) for n in nodes]

    # mathematical testing of a + b = b + a
    ## T1
    ### nodes
    n1_1 = Node(value="+")
    n1_2 = Node(value="a")
    n1_3 = Node(value="b")
    ### and their relations
    n1_1.children = [n1_2, n1_3]
    n1_2.parent_node = n1_1
    n1_3.parent_node = n1_1

    T1 = Tree([n1_1, n1_2, n1_3], root = n1_1)

    ## T2
    ###
    n2_1 = Node(value="+")
    n2_2 = Node(value="a")
    n2_3 = Node(value="b")
    ### and their relations
    n2_1.children = [n2_3, n2_2]
    n2_2.parent_node = n2_1
    n2_3.parent_node = n2_1

    T2 = Tree([n2_1, n2_2, n2_3], root = n2_1)

    # comparing
    treedist, operations, forestdist_dict = tree_edit_distance(T1, T2)
    print(f"difference: {treedist[-1, -1]}")
    print(f"treedist: \n{treedist}")

    # plotting
    T1_nx = T1.to_nx_di_graph()
    T2_nx = T2.to_nx_di_graph()

    plt.figure("T1")
    plot_graph(T1_nx)
    plt.figure("T2")
    plot_graph(T2_nx)

    plt.show()