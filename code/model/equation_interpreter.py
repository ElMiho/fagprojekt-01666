import sympy as sp

from typing import List
from model.tokens import *
from math import *

#####################
# TOKENIZE EQUATION #
#####################

# Helper globals
DIGITS = "0123456789"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvxyz"

SPECIAL_NUMBERS = {
    "Pi": Token(TT_PI),
    "E": Token(TT_E),
    "Phi": Token(TT_PHI),
    "Catalan": Token(TT_CATALAN),
    "EulerGamma": Token(TT_EULERGAMMA),
    "0": Token(TT_ZERO),
    "1": Token(TT_ONE)
}
SPECIAL_NUMBERS_TYPES = [value.t_type for value in SPECIAL_NUMBERS.values()]

UNI_OPERATORS = {
    "Sqrt": Token(TT_SQRT),
    "Sin": Token(TT_SIN),
    "Cos": Token(TT_COS),
    "Tan": Token(TT_TAN),
    "Log": Token(TT_LOG),
}

UNI_OPERATOR_TYPES = [value.t_type for value in UNI_OPERATORS.values()]

BIN_OPERATORS = {
    "+": Token(TT_PLUS),
    "-": Token(TT_MINUS),
    "*": Token(TT_MULTIPLY),
    "/": Token(TT_DIVIDE),
    "^": Token(TT_POW)
}
BIN_OPERATOR_TYPES = [value.t_type for value in BIN_OPERATORS.values()]

OPERATOR_PRECEDENCE = {
    token.t_type: 4 for token in UNI_OPERATORS.values()
}
OPERATOR_PRECEDENCE.update({
    TT_PLUS: 1,
    TT_MINUS: 1,
    TT_MULTIPLY: 2,
    TT_DIVIDE: 2,
    TT_POW: 3
})

#########
# LEXER #
#########

class EquationLexer:
    def __init__(self, text:str):
        self.text = text
        self.position = -1
        self.text_length = len(text)
        self.current_char = None
        self.advance()
    
    ###########################
    # Convert to tokens logic #
    ###########################
    
    # Advances to next character
    def advance(self):
        self.position += 1
        self.current_char = self.text[self.position] if (self.position < self.text_length) else None
    
    def make_tokens(self):
        tokens = []
        
        while (self.current_char != None):
            if self.current_char == " " or self.current_char == "\n":
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self._makeInteger())
            elif self.current_char in ALPHABET:
                func = self._makeFunc()
                if not func: return [] # in case of unrecognized function e.g. PolyGamma
                tokens.append(func)
            elif self.current_char in BIN_OPERATORS:
                token = BIN_OPERATORS[self.current_char]
                if self.current_char == "-" and (not tokens or (tokens and tokens[-1].t_type in (list(BIN_OPERATORS.keys()) + [TT_LEFT_PARENTHESIS]))):
                    # Handle unary minus by prepending a zero
                    tokens.append(Token(TT_ZERO))
                tokens.append(token)
                self.advance()
            elif self.current_char == "(" or self.current_char == "[":
                tokens.append(Token(TT_LEFT_PARENTHESIS))
                self.advance()
            elif self.current_char == ")" or self.current_char == "]":
                tokens.append(Token(TT_RIGHT_PARENTHESIS))
                self.advance()
            elif self.current_char in ALPHABET:
                tokens.append(self._makeFunc())
            else:
                print(f"ERROR unrecognised token: {self.current_char},  position: {self.position}")
                return []
        
        return tokens
    
    def _makeInteger(self):
        res = ""
        res2 = ""

        while self.current_char and self.current_char in DIGITS:
            res += self.current_char
            self.advance()
        
        if self.current_char == "/":
            self.advance()
            while self.current_char in DIGITS:
                res2 += self.current_char
                self.advance()
        
        if res2:
            return Token(TT_RATIONAL, res + "/" + res2)
        else:
            if res in SPECIAL_NUMBERS:
                return SPECIAL_NUMBERS[res]
            return Token(TT_INTEGER, res)
    
    def _makeFunc(self):
        res = ""
        
        while self.current_char and self.current_char in ALPHABET:
            res += self.current_char
            self.advance()
        
        # Map to token type and return it
        ## If: pi,e,phi,...
        if res in SPECIAL_NUMBERS:
            return SPECIAL_NUMBERS[res]
        
        ## If: x,y,z,m,n,...
        elif len(res) == 1:
            return Token(TT_VARIABLE, res)
        
        ## If: sqrt,sin,cos,...
        elif res in UNI_OPERATORS:
            return UNI_OPERATORS[res]

        ## If: unknown
        print(f"Unknown token encountered: [{res}]")
        return None

#################################
# EQUATION REPRESENTATION CLASS #
#################################

class Equation:
    def __init__(self, tokenized_equation:list=None, notation:str="infix"):
        self.tokenized_equation = tokenized_equation
        self.notation = notation

    def convertToInfix(self):
        if self.notation == "infix":
            return None
        elif self.notation == "postfix":
            token_list = list(reversed(self.tokenized_equation))
            def _postfixToInfix(idx=0):
                token = token_list[idx]
                if token.t_type == TT_INTEGER or token.t_type == TT_RATIONAL or token.t_type == TT_VARIABLE or token.t_type in SPECIAL_NUMBERS_TYPES:
                    return [token], idx+1
                elif token.t_type in UNI_OPERATOR_TYPES:
                    center, new_idx = _postfixToInfix(idx+1)
                    return [token, Token(TT_LEFT_PARENTHESIS)] + center + [Token(TT_RIGHT_PARENTHESIS)], new_idx
                elif token.t_type in BIN_OPERATOR_TYPES:
                    right, new_idx = _postfixToInfix(idx+1)
                    left, new_idx = _postfixToInfix(new_idx)
                    return [Token(TT_LEFT_PARENTHESIS)] + left + [token] + right + [Token(TT_RIGHT_PARENTHESIS)], new_idx
            
            infix_token_list, _ = _postfixToInfix()
            self.tokenized_equation = infix_token_list
            self.notation = "infix"

    def convertToPostfix(self):
        if self.notation == "infix":
            token_list = self.tokenized_equation
            operator_stack = []
            output_queue = []
            #################
            # Shunting yard #
            #################
            for token in token_list:
                # Can be uncommented for debugging purposes
                # print("#########\n", token, operator_stack, "\n########")
                if token.t_type == TT_VARIABLE or token.t_type == TT_INTEGER or token.t_type == TT_RATIONAL or token.t_type in SPECIAL_NUMBERS_TYPES:
                    output_queue.append(token)
                elif token.t_type in UNI_OPERATOR_TYPES or token.t_type in BIN_OPERATOR_TYPES:
                    while operator_stack and self._operatorPrecedenceComparison(operator_stack[-1], token) in [0,1]:
                        top_token = operator_stack.pop()
                        output_queue.append(top_token)
                    operator_stack.append(token)
                elif token.t_type == TT_LEFT_PARENTHESIS:
                    operator_stack.append(token)
                elif token.t_type == TT_RIGHT_PARENTHESIS:
                    top_token = operator_stack.pop()
                    while operator_stack and top_token.t_type != TT_LEFT_PARENTHESIS:
                        output_queue.append(top_token)
                        top_token = operator_stack.pop()
                else:
                    print(f"Error, unknown token: [{token}] --- could not convert equation")
                    return None
            
            while operator_stack:
                top_token = operator_stack.pop()
                output_queue.append(top_token)
            
            self.notation = "postfix"
            self.tokenized_equation = output_queue
        
        elif self.notation == "postfix":
            return None

                
    def convertToPrefix(self):
        pass

    def getFloatValue(self):
        equation_copy = Equation(self.tokenized_equation, self.notation)
        #ensure notation is postfix
        if equation_copy.notation != "postfix":
            equation_copy.convertToPostfix()
        assert equation_copy.notation == "postfix"
        
        operand_stack = []
        for token in equation_copy.tokenized_equation:
            if token.t_type == token.t_type in SPECIAL_NUMBERS_TYPES:
                token = self._assignValueToConstant(token)
            if token.t_type == TT_INTEGER or token.t_type == TT_RATIONAL or token.t_type == TT_VARIABLE or token.t_type in SPECIAL_NUMBERS_TYPES:
                if token.t_value == None or isinstance(token.t_value, str): #If a variable has no value assigned, float cannot be calculated
                    return "cannot evaluate"
                operand_stack.append(token)
            elif token.t_type in UNI_OPERATOR_TYPES:     
                operand = operand_stack.pop()
                operand_stack.append(self._applyUniOperator(token, operand))
            elif token.t_type in BIN_OPERATOR_TYPES:
                operand_2 = operand_stack.pop()
                operand_1 = operand_stack.pop()
                operand_stack.append(self._applyBinOperator(token, operand_1, operand_2))
            else:
                print(f"Error, unknown token: [{token}] --- could not get float value")
                return None
        assert len(operand_stack) == 1
        return operand_stack[0].t_value

    def _assignValueToConstant(self, token):
        if token.t_type in SPECIAL_NUMBERS_TYPES:
            match token.t_type:
                case "TT_PI":
                    token.t_value = pi
                case "TT_E":
                    token.t_value = e
                case "TT_ZERO":
                    token.t_value = 0
                case "TT_ONE":
                    token.t_value = 1
                case "TT_PHI":
                    token.t_value = 1.6180339887498948482045
                case "TT_EULERGAMMA":
                    token.t_value = 0.577215664901532860606
                case "TT_CATALAN":
                    token.t_value = 0.9159655941772190150546
                case _: #default
                    token.t_value = None
        return token

    def _applyUniOperator(self, operator, operand):
        output = f"{operator.t_type}({operand.t_value})"
        return (Token(TT_RATIONAL, eval(output, {
            'TT_SIN': sin,
            'TT_COS': cos,
            'TT_TAN': tan,
            'TT_SQRT': sqrt,
            'TT_LOG': log
        })))
    
    def _applyBinOperator(self, operator, operand_1, operand_2):
        operator_string = ""
        match operator.t_type:
            case "TT_PLUS":
                operator_string = "+"
            case "TT_MINUS":
                operator_string = "-"
            case "TT_MULTIPLY":
                operator_string = "*"
            case "TT_DIVIDE":
                operator_string = "/"
            case "TT_POWER":
                operator_string = "**"
            case _: #default
                operator_string = ""
        output = f"{operand_1.t_value}{operator_string}{operand_2.t_value}"
        return Token(TT_RATIONAL, eval(output))
    
    def _operatorPrecedenceComparison(self, operator_1: Token, operator_2: Token):
        """Compares operators in input for higher precedence
        
        Args:
            operator_1 (Token)
            operator_2 (Token)
            
        Returns:
            -1 if either of operator_1 and operator_2 has no precedence or is not an operator
            0 if operator_1 has higher precedence than operator_2
            1 if operator_1 has equal precedence as operator_2
            2 if operator_1 has less precedence than operator_2
        """
        if not operator_1.t_type in OPERATOR_PRECEDENCE or not operator_2.t_type in OPERATOR_PRECEDENCE:
            return -1
        
        # The logicoutput_queue
        if OPERATOR_PRECEDENCE[operator_1.t_type] > OPERATOR_PRECEDENCE[operator_2.t_type]:
            return 0
        elif OPERATOR_PRECEDENCE[operator_1.t_type] == OPERATOR_PRECEDENCE[operator_2.t_type]:
            return 1
        else:
            return 2
        
        
    @classmethod
    def makeEquationFromString(cls, equation:str, notation:str = "infix"):
        """Makes equation instance from string
        
        Args:
            equation (str): the equation to use in string format
            notation (str): the notation used in the string equation
        
        Returns:
            Instance of the Equation class initialized from equation
        """
        lexer = EquationLexer(equation)
        tokenized_equation = lexer.make_tokens()
        return cls(tokenized_equation, notation)


#equation = Equation.makeEquationFromString("-Sin(2-EulerGamma)+a/3+(-7/3*2 + Pi^2)-2")

