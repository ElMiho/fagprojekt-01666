import linecache

from argparse import Namespace
import argparse

args = Namespace(
    # Data information
    inputs_file = ".././data/expressions-1000.txt",
    targets_file = ".././data/answers-1000.txt",
    
    # Model information
    model_save_dir = ".././model_checkpoints",

    # Training information
    num_epochs=1e+3,
    use_cuda=True,

    # Other
    seed=628
)

###########
# DATASET #
###########
class SumDataset:
    pass

# extracting the n'th line (is 0-indexed)
n = 4
input_line = linecache.getline(args.inputs_file, n)
target_line = linecache.getline(args.targets_file, n)
print(input_line)
print(target_line)








