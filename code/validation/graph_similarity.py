import networkx as nx
import sys
import matplotlib.pyplot as plt

# algorithm already implemented in zss
from zss import simple_distance, Node

# SET PATH FOR OSX USERS
if sys.platform == 'darwin':
    sys.path.append('../code')

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

def weisfeiler_leman_1D():
    pass

def wiener_index(G: nx.Graph):
    """
    Input: A graph G
    Output: Computes the Wiener index
    """
    nodes = G1.nodes
    number_of_shortest_paths = 0
    sum = 0
    for v_i in nodes:
        for v_j in nodes:
            d = nx.dijkstra_path_length(G, v_i, v_j)
            if d > 0: number_of_shortest_paths += 1
            sum += d

    return sum/number_of_shortest_paths

# old
def graph_edit_distance(G1: nx.Graph, G2: nx.Graph):
    """
    Inputs: Two graphs G1 and G2
    Outputs: The steps to go from G1 to G2

    Comments and notation:
    (u -> v): u in G1.nodes replaces v in G2.nodes
    (u -> eps): deletion of u on G1.nodes
    (eps -> v): insertion of v in G2.nodes

    all of these are modelled as tuples e.g. (u, v)
    """
    # INIT
    V1 = G1.nodes
    V2 = G2.nodes
    open = set()
    for w in range(len(V2)):
        open.add((0, w))
    open.add((1, "eps"))

    print(f"open: {open}")

## using zss
A = (
    Node("/")
    .addkid(Node("Z"))
    .addkid(Node("^")
        .addkid(Node("pi"))
        .addkid(Node("Z"))
    )
)
    
def insert(G: nx.Graph):
    pass

if __name__ == '__main__':
    print(f"Wiener index of G1: {wiener_index(G1)}")
    print(f"Wiener index of G2: {wiener_index(G2)}")

    graph_edit_distance(G1, G2)

    print(simple_distance(A, A))

    plt.figure(1)
    plot_graph(G1)

    plt.figure(2)
    plot_graph(G2)

    plt.show()