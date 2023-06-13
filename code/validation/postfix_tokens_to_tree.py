import sys

# SET PATH FOR OSX USERS
if sys.platform == 'darwin':
    sys.path.append('../code')


from model import equation_interpreter
from validation.tree_edit_distance import Node, Tree, tree_edit_distance, plot_graph
import matplotlib.pyplot as plt
binary_operators = [
    "TT_PLUS",
    "TT_MINUS",
    "TT_MULTIPLY",
    "TT_DIVIDE", 
    "TT_POW"
]

#def __init__(self, id = None, parent_node = None, value = None, children = []):
    
def generate_nodes_from_postfix(tokens): #Tokens should be listed in postfix!
    
    def nodes_as_a_list(stack):
        nodes = []
        def list_of_kids(node):
            for kid in node.children:
                nodes.append(kid)
                list_of_kids(kid)
        
        root = stack.pop()
        nodes.append(root)
        list_of_kids(root) 
        return nodes
        
    stack = []
    for token in tokens:
        if token.t_type not in binary_operators:
            #new node
            node = Node(value = token.t_type)
            stack.append(node)
        else:
            #new node
            node = Node(value = token.t_type)
            
            #pop stack
            left_sub_tree = stack.pop()
            right_sub_tree = stack.pop()
            
            #assign parrent
            left_sub_tree.parent_node = node
            right_sub_tree.parent_node = node
            
            #assign kids
            node.children = [left_sub_tree, right_sub_tree]
            
            #push to stack
            stack.append(node)
    return nodes_as_a_list(stack)



    

if __name__ == "__main__":
    # TREE 1
    equation = equation_interpreter.Equation.makeEquationFromString("4+Pi/(3+2)")
    equation.convertToPostfix()
    tokens = equation.tokenized_equation
    nodes = generate_nodes_from_postfix(tokens)
    T1 = Tree(nodes = nodes, root = nodes[0])
    
    # TREE 2
    equation2 = equation_interpreter.Equation.makeEquationFromString("Pi/(3+2)+4")
    equation2.convertToPostfix()
    tokens2 = equation2.tokenized_equation
    nodes2 = generate_nodes_from_postfix(tokens2)
    T2 = Tree(nodes = nodes2, root = nodes2[0])
    
    treedist, operations, forestdist_dict = tree_edit_distance(T1, T2)
    print(treedist[-1,-1])
    
    plt.figure(1)
    T1_nx = T1.to_nx_di_graph()
    plot_graph(T1_nx)

    plt.figure(2)
    T2_nx = T2.to_nx_di_graph()
    plot_graph(T2_nx)
    
    plt.show()
    