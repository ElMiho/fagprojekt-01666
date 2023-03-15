# Imports
import torch

from torch.utils.data import Dataset, DataLoader

import linecache
import argparse
import json
import os

from model.equation_interpreter import Equation

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
    def __init__(self, inputs_file:str=config["inputs_file"], targets_file:str=config["targets_file"]) -> None:
        """Data initialization
        
        Args:
            inputs_file (str): path to the file which contains the input data for our NN
            targets_file (str): path to the file which containts the target data for our NN
        """
        self.inputs_file = inputs_file
        self.targets_file = targets_file
        self.dataset_size = sum(1 for i in open(inputs_file, 'rb')) - 1
        self.max_seq_length = max([linecache.getline(args.inputs_file, index).split("\n")[0] for index in range(self.dataset_size)])

    def __getitem__(self, index:int) -> torch.LongTensor:
        # Get corresponding input and target
        input_line = linecache.getline(args.inputs_file, index).split("\n")[0]
        target_line = linecache.getline(args.targets_file, index).split("\n")[0]

        # Transform input and target from string to torch tensor
        input_idx_tensor = torch.LongTensor(json.loads(input_line))
        target_idx_tensor = torch.LongTensor(json.loads(target_line))
        
        return (
            input_idx_tensor,
            target_idx_tensor
        )
    
    def __len__(self) -> int:
        return self.dataset_size
    
    def __repr__(self) -> str:
        return f"<SumDataset(size={len(self)})>"

if config.verbose:
    print(f"Initializing dataset...")
dataset = SumDataset(
    inputs_file=config.inputs_file,
    targets_file=config.targets_file
)
if config.verbose:
    print(f"Dataset `{dataset}` initialized!")





