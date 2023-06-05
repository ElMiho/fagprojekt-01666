import networkx as nx
import sys
import matplotlib.pyplot as plt

# SET PATH FOR OSX USERS
if sys.platform == 'darwin':
    sys.path.append('../code')

# two graphs
## pi^Z / Z
G1 = nx.Graph()
G1.add_node(1, symbol = "/")
G1.add_node(2, symbol = "Z")
G1.add_node(3, symbol = "^")
G1.add_node(4, symbol = "pi")
G1.add_node(5, symbol = "Z")

G1.add_edge(1, 2)
G1.add_edge(1, 3)
G1.add_edge(3, 4)
G1.add_edge(3, 5)

## pi^Z / Z + log(Z)
G2 = nx.Graph()
G2.add_node(1, symbol = "+(1)")
G2.add_node(2, symbol = "log(2)")
G2.add_node(3, symbol = "Z(3)")

G2.add_node(4, symbol = "/(4)")
G2.add_node(5, symbol = "^(5)")
G2.add_node(6, symbol = "pi(6)")
G2.add_node(7, symbol = "Z(7)")
G2.add_node(8, symbol = "Z")

G2.add_edge(1, 2)
G2.add_edge(1, 4)

G2.add_edge(2, 3)

G2.add_edge(4, 5)
G2.add_edge(4, 8)

G2.add_edge(5, 6)
G2.add_edge(5, 7)

# Doesn't plot but rather saves the necessary commands
def plot_graph(G):
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'symbol')
    nx.draw(G, labels=labels)

def weisfeiler_leman_1D():
    pass

def wiener_index(G: nx.Graph):
    nodes = G1.nodes
    number_of_shortest_paths = 0
    sum = 0
    for v_i in nodes:
        for v_j in nodes:
            d = nx.dijkstra_path_length(G, v_i, v_j)
            if d > 0: number_of_shortest_paths += 1
            sum += d

    return sum/number_of_shortest_paths


if __name__ == '__main__':
    print(f"Wiener index of G1: {wiener_index(G1)}")
    print(f"Wiener index of G2: {wiener_index(G2)}")

    plt.figure(1)
    plot_graph(G1)

    plt.figure(2)
    plot_graph(G2)

    plt.show()