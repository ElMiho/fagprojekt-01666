from tqdm import tqdm
import linecache
import argparse
from tqdm import tqdm
import os


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--collection_dir", type=str, help="Whether to print intermediate steps, default ./random-answers-partition-8-4-2023", default="./random-answers-partition-8-4-2023")
args = parser.parse_args()

hm = set()

# If args.collection_dir is a real folder
if args.collection_dir and os.path.exists(args.collection_dir):

    # Write txt files with same name as folder - the monolith file
    for file in tqdm(os.listdir(args.collection_dir)):
        for datapoint in open(os.path.join(args.collection_dir, file)):
            hm.add(datapoint)

with open(args.collection_dir.split(".")[0] + "_unique.txt", "w") as f:
    f.write("".join(list(hm)))
f.close()

