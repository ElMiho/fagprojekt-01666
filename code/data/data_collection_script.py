from tqdm import tqdm
import linecache
import argparse
import os


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--collection_dir", type=str, help="Whether to print intermediate steps, default ./random-answers-partition-7", default="./random-answers-partition-7")
args = parser.parse_args()

# If args.collection_dir is a real folder
if args.collection_dir and os.path.exists(args.collection_dir):

    # Write txt files with same name as folder - the monolith file
    with open(args.collection_dir + ".txt", "w") as f_outer_answers:
        with open(args.collection_dir.replace("answers", "expressions") + ".txt", "w") as f_outer_expressions:
        
            # For every file in the folder
            for filename in tqdm(os.listdir(args.collection_dir), desc=f"Writing monolith files from `{args.collection_dir}`"):
                file = os.path.join(args.collection_dir, filename)
                dataset_size = sum(1 for i in open(file, 'rb'))

                for index in range(1,dataset_size+1):
                    line = linecache.getline(file, index)
                    answer = line.split("}, ")[-1].split("}")[0] + "\n"
                    expression = "}, ".join(line.split("}, ")[:-1]) + "}\n"
                    f_outer_answers.write(answer)
                    f_outer_expressions.write(expression)

        # Close files
        f_outer_expressions.close()
    f_outer_answers.close()


