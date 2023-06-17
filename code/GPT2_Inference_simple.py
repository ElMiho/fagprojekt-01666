# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:09:14 2023

@author: jonas
"""

import torch

from transformers import AutoModelForCausalLM, pipeline, GPT2LMHeadModel, GPT2Tokenizer

import numpy as np
import sympy as sp

from model.tokens import Token, TOKEN_TYPE_EXPRESSIONS, TOKEN_TYPE_ANSWERS
from model.equation_interpreter import Equation
from model.vocabulary import Vocabulary
from model.tokens import Token

from datasets import disable_caching
disable_caching()


# Create a combined vocabulary
vocabulary = Vocabulary.construct_from_list(TOKEN_TYPE_EXPRESSIONS + TOKEN_TYPE_ANSWERS)
vectorized_sample = vocabulary.vectorize(["#", "/", "0", "-1", "[SEP]", "TT_INTEGER"])
vectorized_sample, [vocabulary.getToken(idx) for idx in vectorized_sample]

# Global variables
model_name = "JustSumAI"
project_name = "JustSumAI"
repo_name = f"{model_name}_cleaned_gpt2_data"

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

input_text = "This is my input sequence."
input_ids = tokenizer.encode(input_text, return_tensors='pt')
input_ids, input_ids.size()

vocabulary.end_seq_index

model = GPT2LMHeadModel.from_pretrained(f"Dragonoverlord3000/{model_name}", force_download=True, revision="6512ca7619eafd2da815379268c73c4382b8d3a1")



def neural_network(input_tokens: list):
    test_example_ids = torch.LongTensor([vocabulary.vectorize(input_tokens)[:-1] + [vocabulary.separator_index]])
    out = model.generate(test_example_ids, 
                     eos_token_id=vocabulary.end_seq_index, 
                     pad_token_id=vocabulary.mask_index)
    return [Token(vocabulary.getToken(o.item())) for o in out[0]]

if __name__ == "__main__":
    output = neural_network(["#","/","0","0","0"])