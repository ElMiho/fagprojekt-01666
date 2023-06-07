import networkx as nx
import sys
import matplotlib.pyplot as plt

# algorithm already implemented in zss
# from zss import zss_comapre.simple_distance, zss_simple_tree.Node, distance
import zhang_shasha.zss.compare as zss_compare
import zhang_shasha.zss.simple_tree as zss_simple_tree

# SET PATH FOR OSX USERS
if sys.platform == 'darwin':
    sys.path.append('../code')

binary_operators = [
    "TT_PLUS",
    "TT_MINUS",
    "TT_MULTIPLY",
    "TT_DIVIDE", 
    "TT_POW"
]

basis_functions = [
    "TT_SQRT",
    "TT_SIN",
    "TT_COS",
    "TT_TAN", 
    "TT_LOG"
]

# two graphs
## pi^Z / Z
G1 = nx.Graph()
G1.add_node(0, symbol = "/(0)")
G1.add_node(1, symbol = "Z(1)")
G1.add_node(2, symbol = "^(2)")
G1.add_node(3, symbol = "pi(3)")
G1.add_node(4, symbol = "Z(4)")

G1.add_edge(0, 1)
G1.add_edge(0, 2)
G1.add_edge(2, 3)
G1.add_edge(2, 4)


## pi^Z / Z + log(Z)
G2 = nx.Graph()
G2.add_node(0, symbol = "+(1)")
G2.add_node(1, symbol = "log(2)")
G2.add_node(2, symbol = "Z(3)")

G2.add_node(3, symbol = "/(4)")
G2.add_node(4, symbol = "^(5)")
G2.add_node(5, symbol = "pi(6)")
G2.add_node(6, symbol = "Z(7)")
G2.add_node(7, symbol = "Z")

G2.add_edge(0, 1)
G2.add_edge(0, 3)

G2.add_edge(1, 2)

G2.add_edge(3, 4)
G2.add_edge(3, 7)

G2.add_edge(4, 5)
G2.add_edge(4, 6)

# Doesn't plot but rather saves the necessary commands
def plot_graph(G):
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'symbol')
    nx.draw(G, labels=labels)

## using zss
A = (
    zss_simple_tree.Node("/")
    .addkid(zss_simple_tree.Node("Z"))
    .addkid(zss_simple_tree.Node("^")
        .addkid(zss_simple_tree.Node("pi"))
        .addkid(zss_simple_tree.Node("Z"))
    )
)

B = (
    zss_simple_tree.Node("+")
    .addkid(A)
    .addkid(zss_simple_tree.Node("log").addkid(zss_simple_tree.Node("Z")))
)

C = (
    zss_simple_tree.Node("+")
    .addkid(
        zss_simple_tree.Node("log").addkid(zss_simple_tree.Node("Z"))
    )
    .addkid(
        A
    )
)

def createGraph(tokens): #Assume tokens are listed in postfix
    tokens = tokens[:-1] # reverse
    graph = zss_simple_tree.Node(tokens[0].pop().t_type)
    def addToGraph(tokens):
        if (len(tokens) == 0):
            return graph
        if tokens[0].t_type in binary_operators:
            
            
    
    addToGraph(tokens)
    pass
    
def insert(G: nx.Graph, node1, node2):
    pass

def delete(G: nx.Graph, node):
    pass

def replace(G: nx.Graph, node1, node2):
    pass

def swap(G: nx.Graph, node1, node2):
    pass


if __name__ == '__main__':
    # print("A vs A")
    # print(zss_compare.simple_distance(A, A))
    # print("A vs B")
    # print(zss_compare.simple_distance(A, B, return_operations=True))
    # print("A vs C")
    # print(zss_compare.simple_distance(A, C, return_operations=True))
    # print("B vs C")
    # print(zss_compare.simple_distance(B, C, return_operations=True))
    
    # insert_cost = lambda node: 1
    # remove_cost = lambda node: 1
    # update_cost = lambda a, b: 1
   
    # print(zss_compare.distance(B, C, zss_simple_tree.Node.get_children, insert_cost, remove_cost, update_cost,return_operations=True))

    # plt.figure(1)
    # plot_graph(G1)

    # plt.figure(2)
    # plot_graph(G2)

    # plt.show()

    print("Comparing X and Y")
    X = (
        zss_simple_tree.Node("/")
        .addkid(zss_simple_tree.Node("a"))
        .addkid(zss_simple_tree.Node("b"))
    )

    Y = (
        zss_simple_tree.Node("/")
        .addkid(zss_simple_tree.Node("b"))
        .addkid(zss_simple_tree.Node("a"))
    )
    print(zss_compare.simple_distance(X, Y, return_operations=True))