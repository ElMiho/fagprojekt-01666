#!/usr/bin/env python
# coding: utf-8

# ### Description
# We model the prediction using a transformer decoder, we train GPT-2 to predict the next token. The trick is, that we let every sentence be of the form:
# \begin{align}
#     &["1","5/2","/","0","-1", "[SEPARATOR]", "<MASK>"] \\ 
#     &--- \textrm{Predicted that} "<MASK>" = "Z" ---> \\
#     &["1","5/2","/","0","-1", "[SEPARATOR]", "Z", "<MASK>"]
# \end{align}
# where "[SEPARATOR]" indicates that we're not dealing with the polynomial anymore, we're dealing with the 'answer'. This process is continued iteratively untill the max-length is reached or the network predicts an EOS token

# In[1]:


# External imports
import torch

# For saving and publishing models and datasets to the huggingface hub
from huggingface_hub import notebook_login, create_repo, login

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DownloadConfig, IterableDataset

from tqdm.notebook import tqdm
import linecache
import time
import json


# In[2]:


login()


# In[3]:


# Domestic imports
from model.tokenize_input import input_string_to_tokenize_expression
from model.tokens import TOKEN_TYPE_ANSWERS, TOKEN_TYPE_EXPRESSIONS
from model.equation_interpreter import Equation
from model.vocabulary import Vocabulary


# In[4]:


# Create a combined vocabulary
vocabulary = Vocabulary.construct_from_list(TOKEN_TYPE_EXPRESSIONS + TOKEN_TYPE_ANSWERS)
vectorized_sample = vocabulary.vectorize(["#", "/", "0", "-1", "[SEP]", "TT_INTEGER"])
vectorized_sample, [vocabulary.getToken(idx) for idx in vectorized_sample]


# In[5]:


model_name = "JustSumAI"
repo_name = f"{model_name}_cleaned_gpt2_data"


# # Dataset creation
# Takes `input_file_answers` and `input_file_expressions` and combines every convertable version (i.e. does not contain PolyGamma or similar) of these, separated by a `"[SEP]"` token

# In[6]:


# Number of evaluation datapoints
num_eval_points = 20


# In[7]:


# Paths to data files
input_file_answers = "./data/answers-1000.txt"
input_file_expressions = "./data/expressions-1000.txt"


# In[8]:


max_length = 0

# Make file for cleaned data
cleaned_gpt2_data_train = "/".join(input_file_answers.split("/")[:-1]) + "/cleaned_gpt2_data_train.txt"
cleaned_gpt2_data_eval = "/".join(input_file_answers.split("/")[:-1]) + "/cleaned_gpt2_data_eval.txt"

# Run through every equation in the input files and delete rows with unknown tokens
dataset_size = sum(1 for _ in open(input_file_answers, 'rb'))

# Open and populate cleaned files and
f_cleaned_train = open(cleaned_gpt2_data_train, "w")
f_cleaned_eval = open(cleaned_gpt2_data_eval, "w")

start_time = time.time()
n_cleaned = 0
flag = False
for line_number in tqdm(range(1,dataset_size+1)):
    # Flag determines whether the data point is an evaluation or a training data point
    flag = flag or (line_number-1) % (dataset_size//num_eval_points) == 0
    
    # Get corresponding equation and expression
    raw_equation = linecache.getline(input_file_answers, line_number)
    raw_expression = linecache.getline(input_file_expressions, line_number)

    # Skip line if aborted error
    if raw_equation == "$Aborted\n": continue

    # Construct equation and convert to postfix
    equation = Equation.makeEquationFromString(raw_equation)
    if not equation.tokenized_equation: continue
    equation.convertToPostfix()
    if equation.notation == "infix": continue

    # Vectorize corresponding answer (without SOS) and expression (without EOS)
    vectorized_answers = vocabulary.vectorize([token.t_type for token in equation.tokenized_equation])[1:]
    vectorized_expressions = vocabulary.vectorize([str(token) for token in input_string_to_tokenize_expression(raw_expression)])[:-1]

    max_length = max(max_length, len(vectorized_expressions) + len(vectorized_answers) + 1)

    if flag:
        flag = False
        f_cleaned_eval.write(f"{json.dumps(vectorized_expressions + [vocabulary.getIndex('[SEP]')] + vectorized_answers)}\n")
    else:
        f_cleaned_train.write(f"{json.dumps(vectorized_expressions + [vocabulary.getIndex('[SEP]')] + vectorized_answers)}\n")

    # Write them to cleaned data file and separate them by [SEP] token
    n_cleaned += 1
    
f_cleaned_train.close()
f_cleaned_eval.close()

print(f"\nStats: \n--- Initial size: {dataset_size} \n--- Time since start: {round(time.time() - start_time, 4)} s \n--- Successes: {n_cleaned} \n--- Fails: {line_number - n_cleaned + 1} \n--- Max length: {max_length}")


# In[9]:


with open(f"./data/metadata.txt", "w") as f:
    f.write(json.dumps({"max_length": max_length}))
f.close()


# As per: https://huggingface.co/docs/datasets/create_dataset
# 
# Make a dataset from an iterator and the push it to the hub

# In[10]:


processed_ds_size = sum(1 for _ in open(cleaned_gpt2_data_train, 'rb')) + sum(1 for _ in open(cleaned_gpt2_data_eval, 'rb'))

# Sanity check
print(processed_ds_size,sum(1 for _ in open(cleaned_gpt2_data_train, 'rb')), sum(1 for _ in open(cleaned_gpt2_data_eval, 'rb')))
assert processed_ds_size == n_cleaned

processed_ds_size


# In[11]:


dataset = load_dataset("/".join(cleaned_gpt2_data_train.split("/")[:-1]), 
             data_files={"train": cleaned_gpt2_data_train.split("/")[-1], "validation": cleaned_gpt2_data_eval.split("/")[-1]})
dataset


# In[12]:


create_repo(repo_name, exist_ok=True, repo_type="dataset")


# In[13]:


dataset.push_to_hub(repo_name)


# In[14]:


dataset = load_dataset(f"Dragonoverlord3000/{repo_name}")


# In[15]:


json.loads(dataset["train"][0]["text"])


# # Training GPT-2 from scratch

# ### Loading the model
# The model is GPT-2 with 1.5 BILLION  parameters!!!

# In[16]:


config = AutoConfig.from_pretrained("gpt2-xl", vocab_size=len(vocabulary), 
                                    bos_token_id=vocabulary.begin_seq_index,
                                    eos_token_id=vocabulary.end_seq_index,
                                    max_length=max_length)
model = AutoModelForCausalLM.from_config(config)
model


# In[17]:


f"GPT-2 (xl) size: {model.num_parameters()/1_000_000:.1f}M parameters"


# In[19]:


# Save newly initialized model - takes a while
model.save_pretrained(f"models/{model_name}", 
                      push_to_hub=True, 
                      organization="Dragonoverlord3000")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




