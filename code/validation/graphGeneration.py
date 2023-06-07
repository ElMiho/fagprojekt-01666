import sys
from zss import simple_distance, Node
from sympy.utilities.iterables import variations

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

equation = equation_interpreter.Equation.makeEquationFromString("3*4/2+1")
equation.convertToPostfix()
tokens = equation.tokenized_equation
equation1 = equation_interpreter.Equation.makeEquationFromString("4*3/2+1)")
equation1.convertToPostfix()
tokens1 = equation.tokenized_equation

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
g1 = generate_graph_from_postfix(tokens1)

def generate_combinations(graph):
    
    combinations = []
    combinations.append(graph.copy())

    

    def recursive_node_search(node):
        if node.children == []:
            return
        else:
            for child in node.children:
                if child.label in plus_mulitply:
                    child0 = child.children[0]
                    child1 = child.children[1]
                    child.children[0] = child1
                    child.children[1] = child0
                    combinations.append(graph.copy())
                    
                recursive_node_search(child)
    recursive_node_search(graph)

    #IF ROOT IS MULT/ADD, FLIP CHILDREN
    if graph.label in plus_mulitply:
        child0 = graph.children[0]
        child1 = graph.children[1]
        graph.children[0] = child1
        graph.children[1] = child0
        combinations.append(graph.copy())
        recursive_node_search(graph)

    return combinations


def getDistance(graph1, graph2):
    combs = generate_combinations(graph1.copy())
    smallest_distance = sys.maxsize
    for c in combs:
        distance = simple_distance(c, graph2)
        if distance < smallest_distance:
            smallest_distance = distance
    return smallest_distance




print(getDistance(g, g1))
