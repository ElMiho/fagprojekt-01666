import argparse
import os


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--collection_dir", type=str, help="Whether to print intermediate steps, e.g. ./random-answers-partition-7", default="")
args = parser.parse_args()

# If args.collection_dir is a real folder
if args.collection_dir and os.path.exists(args.collection_dir):

    # Write a 'txt' file with same name as folder - the monolith file
    with open(args.collection_dir + ".txt", "w") as f_outer:
    
        # For every file in the folder
        for filename in os.listdir(args.collection_dir):
    
            # Write data from file to monolith file
            with open(os.path.join(args.collection_dir, filename), "r") as f_inner:
                f_outer.write(f_inner.read())
            f_inner.close()
        
    f_outer.close()