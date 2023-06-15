from model.tokenize_input import token_input_space

# Token class
class Token:
    def __init__(self, t_type: str, t_value:str = None):
        self.t_type = t_type
        self.t_value = t_value
        
    def __repr__(self):
        return f"---- Type: {self.t_type} \t Value: {self.t_value} ----"

# Tokens - Mathematical expressions
## Variable
TT_VARIABLE = "TT_VARIABLE"

## Specials
TT_LEFT_PARENTHESIS = "TT_LEFT_PARENTHESIS"
TT_RIGHT_PARENTHESIS = "TT_RIGHT_PARENTHESIS"

## Special numbers
TT_PI = "TT_PI"
TT_E = "TT_E"
TT_PHI = "TT_PHI"
TT_CATALAN = "TT_CATALAN"
TT_EULERGAMMA = "TT_EULERGAMMA" # i.e. the euler-mascheroni constant
TT_ZERO = "TT_ZERO"
TT_ONE = "TT_ONE"

## Numbers
TT_INTEGER = "TT_INTEGER"
TT_RATIONAL = "TT_RATIONAL"

## Unary Operations
TT_SQRT = "TT_SQRT"
TT_SIN = "TT_SIN"
TT_COS = "TT_COS"
TT_TAN = "TT_TAN"
TT_LOG = "TT_LOG"

## Binary Operations
TT_PLUS = "TT_PLUS"
TT_MINUS = "TT_MINUS"
TT_MULTIPLY = "TT_MULTIPLY"
TT_DIVIDE = "TT_DIVIDE"
TT_POW = "TT_POW"

unary_operators = {
    TT_SQRT, TT_SIN, TT_COS, TT_TAN, TT_LOG
}

binary_operators = {
    TT_PLUS, TT_MINUS, TT_MULTIPLY, TT_DIVIDE, TT_POW
}

commutative_operators = {
    TT_PLUS, TT_MULTIPLY
}

# Token to latex mapper
token2symbol = {
    TT_LEFT_PARENTHESIS: "(",
    TT_RIGHT_PARENTHESIS: ")",

    ## Special numbers
    TT_PI: r"\pi",
    TT_E: "e",
    TT_PHI: r"\varphi",
    TT_CATALAN: "G",
    TT_EULERGAMMA: r"\gamma", # i.e. the euler-mascheroni constant
    TT_ZERO: "0",
    TT_ONE: "1",

    ## Numbers
    TT_INTEGER: "Z",

    ## Unary Operations
    TT_SQRT: r"\sqrt{\;}",
    TT_SIN: r"\sin",
    TT_COS: r"\cos",
    TT_TAN: r"\tan",
    TT_LOG: r"\log",

    ## Binary Operations
    TT_PLUS: "+",
    TT_MINUS: "-",
    TT_MULTIPLY: r"\times",
    TT_DIVIDE: "/",
    TT_POW: r"^",
    None: None
}


## List of all token types
TOKEN_TYPE_ANSWERS = [
    ## Variable
    TT_VARIABLE,
    ## Specials
    TT_LEFT_PARENTHESIS, TT_RIGHT_PARENTHESIS,
    ## Special numbers
    TT_PI, TT_E, TT_PHI, TT_CATALAN, TT_EULERGAMMA,
    ## Numbers
    TT_INTEGER, TT_RATIONAL, TT_ZERO, TT_ONE,
    ## Unary Operations
    TT_SQRT, TT_SIN, TT_COS, TT_TAN, TT_LOG,
    ## Binary Operations
    TT_PLUS, TT_MINUS, TT_MULTIPLY, TT_DIVIDE, TT_POW
]

# Tokens - Roots
TOKEN_TYPE_EXPRESSIONS = [str(v) for v in token_input_space(-5,5)]

if __name__ == "__main__":
    print(f"TT ANSWERS: {TOKEN_TYPE_ANSWERS}")
    print(f"TT EXPRESSIONS: {TOKEN_TYPE_EXPRESSIONS}")