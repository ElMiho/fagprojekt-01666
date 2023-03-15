# Imports
import torch

from torch.utils.data import Dataset, DataLoader

import linecache
import argparse
import json
import os

from model.equation_interpreter import Equation
from model.vocabulary import vocabulary_answers, vocabulary_expressions, Vocabulary

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

###########
# DATASET #
###########
class SumDataset(Dataset):
    def __init__(self, inputs_file:str=config["inputs_file"], targets_file:str=config["targets_file"],
                 input_vocab:Vocabulary=vocabulary_answers, target_vocab:Vocabulary=vocabulary_expressions) -> None:
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
            self.max_seq_length_target = max(self.max_seq_length_target, len(json.loads(linecache.getline(self.inputs_file, index))))
    
    def __getitem__(self, index:int) -> dict:
        index += 1

        # Get corresponding input and target
        input_line = linecache.getline(self.inputs_file, index)
        target_line = linecache.getline(self.targets_file, index)

        # Transform input and target from string to list
        input_idx_list = json.loads(input_line)
        target_idx_list = json.loads(target_line)
        input_length = len(input_idx_list)

        # Pad lists and split target list to target_x list and target_y list
        input_idx_list.extend([0] * (self.max_seq_length_input - len(input_idx_list)))
        
        target_idx_list_x = target_idx_list[:-1]
        target_idx_list_y = target_idx_list[1:]
        target_idx_list_x.extend([0] * (self.max_seq_length_target - len(target_idx_list_x)))
        target_idx_list_y.extend([0] * (self.max_seq_length_target - len(target_idx_list_y)))

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

if config["verbose"]:
    print(f"Initializing dataset...")
dataset = SumDataset(
    inputs_file=config["inputs_file"],
    targets_file=config["targets_file"]
)
if config["verbose"]:
    print(f"Dataset `{dataset}` initialized!")


print(dataset[0])


