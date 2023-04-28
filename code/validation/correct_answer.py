import sys
import networkx as nx
import matplotlib.pyplot as plt

# SET PATH FOR OSX USERS
if sys.platform == 'darwin':
    sys.path.append('../code')

import model.tokenize_input as tokenize_input
import model.equation_interpreter as equation_interpreter

binary_operators = [
    "TT_PLUS",
    "TT_MINUS",
    "TT_MULTIPLY",
    "TT_DIVIDE", 
    "TT_POW"
]

def equation_string_to_graph(input_string: str):
    eq = equation_interpreter.Equation.makeEquationFromString(input_string)
    eq.convertToPostfix()
    G = nx.Graph()

    token_list = eq.tokenized_equation

    for idx, node in enumerate(reversed(token_list)):
        type = node.t_type
        value = node.t_value
        G.add_node(idx, symbol = type)

    for idx, node in enumerate(reversed(token_list)):
        type = node.t_type
        value = node.t_value

        if type in binary_operators:
            G.add_edge(idx, idx + 1)
            G.add_edge(idx, idx + 2)

    return G

# Doesn't plot but rather saves the necessary commands
def plot_graph(G):
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'symbol')
    nx.draw(G, labels=labels)


if __name__ == '__main__':
    test_string = "Pi^2/6"
    predicted_test_string = "Pi^Z/Z + Pi"

    plt.figure(1)
    G = equation_string_to_graph(test_string)
    plot_graph(G)


    plt.figure(2)
    J = equation_string_to_graph(predicted_test_string)
    plot_graph(J)

    print(nx.graph_edit_distance(
        G, J
    ))

    plt.show()