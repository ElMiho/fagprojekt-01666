from typing import List
from tokens import Token

class Vocabulary:
    def __init__(self, token2index, index2token) -> None:
        self.token2index = token2index
        self.index2token = index2token

    def addToken(self, token: Token) -> int:
        # Don't add existing tokens
        if token in self.token2index:
            return self.token2index[token]
        
        index = len(self.token2index)
        self.token2index[token] = index
        self.index2token[index] = token
        return index

    def getToken(self, index: int) -> Token:
        if not index in self.index2token:
            return -1
        return self.index2token[index]

    def getIndex(self, token: Token) -> int:
        if not token in self.token2index:
            return -1
        return self.token2index[token]

    def vectorize(self, token_list: List[Token]) -> List[Token]:
        index_list = [
            self.getIndex[token] for token in token_list
        ]
        return index_list

    @classmethod
    def construct_from_list(cls, token_types: List[Token]):
        token2index = {token.t_type:idx for idx,token in enumerate(token_types)}
        index2token = {idx:token.t_type for idx,token in enumerate(token_types)}

        return cls(token2index, index2token)



