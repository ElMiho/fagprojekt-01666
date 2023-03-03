from typing import List
from model.tokens import Token, TOKEN_TYPE_ANSWERS

class Vocabulary:
    def __init__(self, token2index, index2token, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>") -> None:
        # special tokens
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.begin_seq_token = begin_seq_token
        self.end_seq_token = end_seq_token
        ## Special token indices
        self.unk_index = self.addToken(unk_token)
        self.mask_index = self.addToken(mask_token)
        self.begin_eq_index = self.addToken(begin_seq_token)
        self.end_seq_index = self.addToken(end_seq_token)

        # converter mappings
        self.token2index = token2index
        self.index2token = index2token

    def addToken(self, token: Token) -> int:
        """
        Args:
            token (Token): the token to add to the vocabulary

        Returns:
            The index of the token that has either just been added 
            or the index of the token if it already exists in the 
            vocabulary
        """
        # Don't add existing tokens
        if token in self.token2index:
            return self.token2index[token]
        
        index = len(self.token2index)
        self.token2index[token] = index
        self.index2token[index] = token
        return index

    def getToken(self, index: int) -> Token:
        """
        Args:
            index (int): the index of the token to get

        Returns:
            The corresponding token
        """

        if not index in self.index2token:
            return -1
        return self.index2token[index]

    def getIndex(self, token: Token) -> int:
        """
        Args:
            token (Token): the token to get the index of

        Returns:
            The corresponding index
        """
        if not token in self.token2index:
            return -1
        return self.token2index[token]

    def vectorize(self, token_list: List[Token]) -> List[Token]:
        """
        Args:
            token_list (List[Token]): list of the tokens to turn into indices

        Returns:
            The correspinding index list
        """
        index_list = [
            self.getIndex[token] for token in token_list
        ]
        return index_list

    @classmethod
    def construct_from_list(cls, token_types: List[str]):
        """
        Args:
            token_types (List[str]): list of token types

        Returns:
            Instance of the vocabulary instantiated with the given tokens
        """

        token2index = {token_type:idx for idx,token_type in enumerate(token_types)}
        index2token = {idx:token_type for idx,token_type in enumerate(token_types)}

        return cls(token2index, index2token)


vocabulary_answers = Vocabulary.construct_from_list(TOKEN_TYPE_ANSWERS)
vocabulary_expressions = None


