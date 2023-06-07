import sys
from zss import simple_distance, Node

binary_operators = [
    "TT_PLUS",
    "TT_MINUS",
    "TT_MULTIPLY",
    "TT_DIVIDE", 
    "TT_POW"
]

plus_mulitply = [
    "TT_PLUS",
    "TT_MULTIPLY"
]

basis_functions = [
    "TT_SQRT",
    "TT_SIN",
    "TT_COS",
    "TT_TAN", 
    "TT_LOG"
]


# SET PATH FOR OSX USERS
if sys.platform == 'darwin':
    sys.path.append('../code')

from model import equation_interpreter

A = (
    Node("TT_DIVIDE")
    .addkid(Node("TT_INTEGER"))
    .addkid(Node("TT_POW")
        .addkid(Node("TT_PI"))
        .addkid(Node("TT_INTEGER"))
    )
)

equation = equation_interpreter.Equation.makeEquationFromString("4+Pi")
equation.convertToPostfix()
tokens = equation.tokenized_equation

def generate_expressions(tokens): #Tokens should be listed in postfix!
    #This does absolutely not work
    stack = [[]]
        

    for token in tokens:
        if token.t_type not in binary_operators:
            for s in stack:
                s.append(token)
        else:
            for idx, s in enumerate(stack):
                op1 = stack[idx].pop()
                op2 = stack[idx].pop()

    return stack


def generate_graph_from_postfix(tokens): #Tokens should be listed in postfix!
    stack = []
    
    for token in tokens:
        if token.t_type not in binary_operators:

            stack.append(Node(token.t_type))
        else:
            operand2 = stack.pop()
            operand1 = stack.pop()
            stack.append(Node(token.t_type, [operand1, operand2]))
    return stack.pop()

g = generate_graph_from_postfix(tokens)

combinations = []


# Recursive function to generate all combinations
def generate_combinations(node):
    # Base case: if node is a leaf, return a list containing only the node
    if node.children == []:
        return [node]

    combinations = []

    # Check if the current node is an operator that needs flipping
    if node.label in plus_mulitply:
        # Generate combinations by flipping the children
        left_combinations = generate_combinations(node.children[0])
        right_combinations = generate_combinations(node.children[1])

        # Generate combinations with all possible permutations
        for left_node in left_combinations:
            for right_node in right_combinations:
                flipped_node = Node(node.label)
                flipped_node.addkid(right_node)
                flipped_node.addkid(left_node)
                combinations.append(Node(node.label, [left_node, flipped_node]))
                combinations.append(Node(node.label, [flipped_node, right_node]))
    else:
        # Generate combinations for the non-operator node
        left_combinations = generate_combinations(node.children[0])
        right_combinations = generate_combinations(node.children[1])

        # Generate combinations by combining the left and right sub-combinations
        for left_node in left_combinations:
            for right_node in right_combinations:
                new_node = Node(node.label)
                new_node.addkid(left_node)
                new_node.addkid(right_node)
                combinations.append(new_node)

    return combinations

combs = generate_combinations(g)


for c in combs:
    print('new comb')
    print(c.children[1])
