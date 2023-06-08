import sys
from zss import simple_distance, Node
#from validation.zhang_shasha.zss.compare import simple_distance
#from validation.zhang_shasha.zss.simple_tree import Node



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


def generate_graph_from_postfix2(tokens): #Tokens should be listed in postfix!
    # DRAFT 
    stacks = [[]]
    
    for token in tokens:
        if token.t_type not in binary_operators:
            for stack in stacks:
                stack.append(Node(token.t_type))
        else:
            for i in range(0, len(stacks)):
                stack = stacks[i]
                operand2 = stack.pop()
                operand1 = stack.pop()
                stack_copy = stack
                stack.append(Node(token.t_type, [operand1, operand2]))
                stacks[i] = stack
                if token.t_type in plus_mulitply:
                    stack_copy.append(Node(token.t_type, [operand2, operand1]))
                    stacks.append(stack_copy)
                    
    return stacks

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

#Test cases
equation = equation_interpreter.Equation.makeEquationFromString("4+Pi/(3+2)")
equation.convertToPostfix()
tokens = equation.tokenized_equation
equation1 = equation_interpreter.Equation.makeEquationFromString("Pi/(2+3)+4")
equation1.convertToPostfix()
tokens1 = equation1.tokenized_equation
g = generate_graph_from_postfix(tokens)
g1 = generate_graph_from_postfix(tokens1)

dist = getDistance(g, g1)
print(dist)