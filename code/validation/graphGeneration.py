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

equation = equation_interpreter.Equation.makeEquationFromString("3 * 4 + 2")
equation.convertToPostfix()
tokens = equation.tokenized_equation

def generate_expressions(tokens): #Tokens should be listed in postfix!
    #This does absolutely not work
    stack = [[]]
    for token in tokens:
        if token.t_type not in binary_operators:
            for sck in stack:
                sck.append(token)
        else:
          stack.append(stack[-1].copy())
          for sck in stack:
              op2 = sck.pop()
              op1 = sck.pop()
              sck.append([op1,token,op2])
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

#g = generate_graph_from_postfix(tokens)
result = generate_expressions(tokens)
for r in result:
    print("RESULT:")
    for t in r:
        print(t.t_type, t.t_value)