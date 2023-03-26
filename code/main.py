# Imports
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tqdm import tqdm
import numpy as np
import linecache
import argparse
import json
import time
import os

from model.equation_interpreter import Equation
from model.vocabulary import Vocabulary
from model.vocabulary import vocabulary_answers as target_vocabulary
from model.vocabulary import vocabulary_expressions as source_vocabulary

from model.model import Model

#########
# SETUP #
#########

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-config", type=str, help="Name of the config file to use (see the configs folder) e.g. `-config config.json`")
parser.add_argument("--verbose", type=bool, help="Whether to print intermediate steps", default=True)
args = parser.parse_args()

# Custom configuration file
if args.config and os.path.exists(f"./configs/{args.config}"):
    config_path = f"./configs/{args.config}"
# Default config file
else:
    config_path = "./configs/default.json"

if args.verbose: 
    print(f"Using configuration: {config_path}")

file = open(config_path, "r")
config = json.load(file)
file.close()


# Setup
device = "cuda" if config["use_cuda"] and torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


###########
# DATASET #
###########
class SumDataset(Dataset):
    def __init__(self, inputs_file:str=config["inputs_file"], targets_file:str=config["targets_file"],
                 input_vocab:Vocabulary=source_vocabulary, target_vocab:Vocabulary=target_vocabulary) -> None:
        """Data initialization
        
        Args:
            inputs_file (str): path to the file which contains the input data for our NN
            targets_file (str): path to the file which containts the target data for our NN
        """
        # Instantiate input + target files and vocabs
        self.inputs_file = inputs_file
        self.targets_file = targets_file
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab

        # Get dataset stats
        self.dataset_size = sum(1 for i in open(inputs_file, 'rb')) - 1
        self.max_seq_length_input = 0
        self.max_seq_length_target = 0
        for index in range(1,self.dataset_size+1):
            self.max_seq_length_input = max(self.max_seq_length_input, len(json.loads(linecache.getline(self.inputs_file, index))))
            self.max_seq_length_target = max(self.max_seq_length_target, len(json.loads(linecache.getline(self.targets_file, index))))
    
    def __getitem__(self, index:int) -> dict:
        # linecache.getline 1-indexes
        index += 1

        # Get corresponding input and target
        input_line = linecache.getline(self.inputs_file, index)
        target_line = linecache.getline(self.targets_file, index)

        # Transform input and target from string to list
        input_idx_list = json.loads(input_line)
        target_idx_list = json.loads(target_line)
        input_length = len(input_idx_list)

        # Pad lists and split target list to target_x list and target_y list
        input_idx_list.extend([self.input_vocab.mask_index] * (self.max_seq_length_input - len(input_idx_list)))
        
        target_idx_list_x = target_idx_list[:-1]
        target_idx_list_y = target_idx_list[1:]
        target_idx_list_x.extend([self.target_vocab.mask_index] * (self.max_seq_length_target - len(target_idx_list_x)))
        target_idx_list_y.extend([self.target_vocab.mask_index] * (self.max_seq_length_target - len(target_idx_list_y)))

        # Convert to pytorch tensor
        input_idx_tensor = torch.LongTensor(input_idx_list)
        target_idx_tensor_x = torch.LongTensor(target_idx_list_x)
        target_idx_tensor_y = torch.LongTensor(target_idx_list_y)

        return {
            "input": input_idx_tensor,
            "input_lengths": input_length,
            "target_x": target_idx_tensor_x,
            "target_y": target_idx_tensor_y
        }
    
    def __len__(self) -> int:
        return self.dataset_size
    
    def __repr__(self) -> str:
        return f"<SumDataset(size={len(self)})>"

def generate_nmt_batches(dataset):
    """A generator function which wraps the PyTorch DataLoader. The NMT Version """
    dataloader = DataLoader(dataset, 
        batch_size=config["batch_size"],    # samples data into collections
        shuffle=config["shuffle"],          # shuffles the indices
        drop_last=config["drop_last"]       # drop the last batch if len(data) does not divide batch_size
    )

    for batch_index, data_dict in enumerate(dataloader):
        lengths = data_dict['input_lengths'].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()
        
        out_data_dict = {}
        for name in data_dict:
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield batch_index, out_data_dict 

if args.verbose:
    print(f"Initializing dataset...")
dataset = SumDataset(
    inputs_file=config["inputs_file"],
    targets_file=config["targets_file"]
)
if args.verbose:
    print(f"Dataset `{dataset}` initialized!")
    dataset_stats = {
        "inputs_file": dataset.inputs_file,
        "targets_file": dataset.targets_file,
        "input_vocab": dataset.input_vocab,
        "target_vocab": dataset.target_vocab,

        # Get dataset stats
        "dataset_size": len(dataset),
        "max_seq_length_input": dataset.max_seq_length_input,
        "max_seq_length_target": dataset.max_seq_length_target
    }
    print(f"Dataset stats: {dataset_stats}")


# 
# Structure and code is loosely modeled based on the following jupyter notebook: https://github.com/delip/PyTorchNLPBook/blob/master/chapters/chapter_8/8_5_NMT/8_5_NMT_No_Sampling.ipynb
# Article describing pytorch's built in RNN module (i.e. sizes and such): https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-rnn-cb6ebc594677
# 


################
# HELPER STUFF #
################

# For recording the training history
train_state = {
        "stop_early": False,
        "early_stopping_step": 0,
        "early_stopping_best_val": 1e+8,
        "learning_rate": config["learning_rate"],
        "epoch_index": 0,
        "train_loss": [],
        "train_acc": [],
        "model_filename": config["model_filename"]
    }

def normalize_sizes(y_pred, y_true):
    """Normalizes tensor sizes
    
    Args:
        y_pred: the output of the model
            If 3D tensor, reshape to 2D tensor (matrix)
        y_true: the target predictions
            If 2D tensor (matrix), reshape to 1D tensor (vector)
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true

def compute_accuracy(y_pred, y_true, mask_index):
    # Make sure y_pred has 2D shape and y_true a 1D shape
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    # torch.tensor.max(dim=x) returns a tensor of maximum values and their corresponding indices
    _, y_pred_indices = y_pred.max(dim=1)
    
    # Find every non-mask index that was correctly predicted
    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()
    
    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100

####################
# INITIALIZE MODEL #
####################

# The model
model = Model(
    source_vocab_size=len(source_vocabulary), source_embedding_size=config["embedding_size"],
    target_vocab_size=len(target_vocabulary), target_embedding_size=config["embedding_size"],
    encoding_size=config["rnn_hidden_size"], target_bos_index=source_vocabulary.begin_seq_index,
    max_seq_length=dataset.max_seq_length_target
)
model = model.to(device)

if args.verbose:
    print(f"Using model `{config['model_filename']}` with architecture:")
    print(model)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# It is framed as a classification problem - predict the next word
cross_entropy = nn.CrossEntropyLoss(ignore_index=target_vocabulary.mask_index)

# Get loss of prediction
def sequence_loss(y_pred, y_true, mask_index=target_vocabulary.mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return cross_entropy(y_pred, y_true)

#################
# TRAINING LOOP #
#################

## Note: tqdm just adds a progress bar to the training loop
epoch_iterator = tqdm(range(config["num_epochs"]), desc=f"Running loss: ---, Running acc: ---")
for epoch in epoch_iterator:
    train_state["epoch_index"] = epoch

    running_loss = 0
    running_acc = 0
    dataloader = generate_nmt_batches(dataset)
    # Makes sure dropout is used
    model.train()

    for batch_index, batch_dict in dataloader:
        # 1. zero the gradients
        optimizer.zero_grad()

        # 2. predict the output
        y_pred = model(
            batch_dict["input"],
            batch_dict["input_lengths"],
            batch_dict["target_x"]
        )

        # 3. compute the loss
        loss = sequence_loss(y_pred, batch_dict["target_y"], target_vocabulary.mask_index)

        # 4. use loss to calculate gradient
        loss.backward()

        # 5. optimizer
        optimizer.step()
        
        # Calculate the running loss and running accuracy
        running_loss += (loss.item() - running_loss) / (batch_index + 1)
        acc_t = compute_accuracy(y_pred, batch_dict["target_y"], target_vocabulary.mask_index)
        running_acc = (acc_t - running_acc) / (batch_index + 1)

    train_state["train_loss"].append(running_loss)
    train_state["train_acc"].append(running_acc)

    # Display running loss and accuracy
    epoch_iterator.set_description(f"Running loss: {round(running_loss, 4)}, Running acc: {round(running_acc, 4)}")
    # Save model
    torch.save(model.state_dict(), train_state['model_filename'])


#############
# INFERENCE #
#############
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
], dtype=torch.int32).to(device)
test_pred = model(
    test_tensor,
    torch.LongTensor([len(test_tensor[0]) for _ in range(len(test_tensor))]).to(device),
    target_sequence=None
)

print(f"Test expression: {test_expression}")
print(f"Predicted shape: {test_pred.shape}")
print(f"Predicted value: {sentence_from_indices(to_indices(test_pred[0]), target_vocabulary)}")














