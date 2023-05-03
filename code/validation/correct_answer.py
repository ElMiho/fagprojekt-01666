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

basis_functions = [
    "TT_SQRT",
    "TT_SIN",
    "TT_COS",
    "TT_TAN", 
    "TT_LOG"
]

def construct_graph_helper(token_list, idx):
    if token_list[idx] in basis_functions:

def equation_string_to_graph(input_string: str):
    eq = equation_interpreter.Equation.makeEquationFromString(input_string)
    eq.convertToPostfix()
    G = nx.DiGraph()

    token_list = eq.tokenized_equation
    token_list.reverse()

    for idx, node in enumerate(token_list):
        type = node.t_type
        value = node.t_value
        G.add_node(idx, symbol = type)

    for idx, node in enumerate(token_list):
        type = node.t_type
        value = node.t_value

        if type in basis_functions:
            G.add_edge(idx, idx + 1)

        if type in binary_operators:
            next_1 = token_list[idx + 1]
            next_2 = token_list[idx + 2]

            if next_1.t_type in basis_functions:
                counter = 0
                pass
            
            G.add_edge(idx, idx + 1)
            G.add_edge(idx, idx + 2)

    return G

# Doesn't plot but rather saves the necessary commands
def plot_graph(G):
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'symbol')
    nx.draw(G, labels=labels)

if __name__ == '__main__':
    test_string = "Pi^Z/Z"
    predicted_test_string = "Pi^Z/Z + Log[Z]"

    eq1 = equation_interpreter.Equation.makeEquationFromString(test_string)
    eq1.convertToPostfix()
    print(f"eq1: {eq1.tokenized_equation}")

    eq2 = equation_interpreter.Equation.makeEquationFromString(predicted_test_string)
    eq2.convertToPostfix()
    print(f"eq2: {eq2.tokenized_equation}")

    plt.figure(1)
    G = equation_string_to_graph(test_string)
    plot_graph(G)


    plt.figure(2)
    J = equation_string_to_graph(predicted_test_string)
    plot_graph(J)

    print(G.nodes.data())
    print(J.nodes.data())

    print(nx.graph_edit_distance(
        G, J, roots = (0, 0)
    ))

    # test_2 = "Log[Z + R]"
    # G = equation_string_to_graph(test_2)
    # plt.figure(1)
    # plot_graph(G)

    plt.show()