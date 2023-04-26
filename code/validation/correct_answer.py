# Copied a lot of stuff from inference.ipynb 
# due to the model from main.py didn't give correct answers

import sys
#SET PATH FOR OSX USERS
if sys.platform == 'darwin':
    sys.path.append('../code')


import torch

import sympy as sp

from model.vocabulary import vocabulary_answers as target_vocabulary
from model.vocabulary import vocabulary_expressions as source_vocabulary
from model.equation_interpreter import Equation
from model.tokens import Token

import json


from model.model import Model

config_path = "./configs/default.json"
print(f"Using configuration: {config_path}")

file = open(config_path, "r")
config = json.load(file)
file.close()


device = "cuda" if torch.cuda.is_available() else "cpu"


model = Model(
    source_vocab_size=len(source_vocabulary)-1, source_embedding_size=config["embedding_size"],
    target_vocab_size=len(target_vocabulary)-1, target_embedding_size=config["embedding_size"],
    encoding_size=config["rnn_hidden_size"], target_bos_index=target_vocabulary.begin_seq_index,
    max_seq_length=185
).to(device)

checkpoint = torch.load("model_3.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)


model.eval()


def to_indices(scores):
    _, indices = torch.max(scores, dim=1)
    return indices

def sentence_from_indices(indices, vocab, strict=True):
    out = []
    for index in indices:
        index = index.item()
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            return " ".join(out)
        else:
            out.append(vocab.getToken(index))
    return " ".join(out)


# n^2 test
test_expression = ["#", "/", "0", "0"]
test_tensor = torch.tensor([
    source_vocabulary.vectorize(test_expression) for _ in range(config["batch_size"])
], dtype=torch.int32)
test_pred = model(
    test_tensor,
    torch.LongTensor([len(test_tensor[0]) for _ in range(len(test_tensor))]),
    target_sequence=None
)

prediction = sentence_from_indices(to_indices(test_pred[0]), target_vocabulary)

print(f"Test expression: {test_expression}")
print(f"Predicted shape: {test_pred.shape}")
print(f"Predicted value: {prediction}")


token_list = [Token(t_type) for t_type in prediction.split(" ")]
predicted_equation = Equation(token_list, notation="postfix")
print(
    predicted_equation.getMathmetaicalNotation()
)