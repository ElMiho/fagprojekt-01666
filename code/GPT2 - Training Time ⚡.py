#!/usr/bin/env python
# coding: utf-8

# External imports
import torch

from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, Dataset
#from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

import datasets
import transformers
from transformers import AutoModel, set_seed, get_scheduler, AutoModelForCausalLM
from huggingface_hub import Repository, create_repo, login
from datasets import load_dataset
from accelerate import Accelerator

from argparse import Namespace
import logging
import json
import os

# Domestic imports
from model.tokenize_input import input_string_to_tokenize_expression
from model.tokens import TOKEN_TYPE_ANSWERS, TOKEN_TYPE_EXPRESSIONS
from model.equation_interpreter import Equation
from model.vocabulary import Vocabulary
from model.tokens import Token

# Create a combined vocabulary
vocabulary = Vocabulary.construct_from_list(TOKEN_TYPE_EXPRESSIONS + TOKEN_TYPE_ANSWERS)
vectorized_sample = vocabulary.vectorize(["#", "/", "0", "-1", "[SEP]", "TT_INTEGER"])
vectorized_sample, [vocabulary.getToken(idx) for idx in vectorized_sample]

# Global variables
model_name = "JustSumAI"
project_name = "JustSumAI"
repo_name = f"{model_name}_cleaned_gpt2_data"

login()

# # Setup
with open("./data/metadata.txt", "r") as f:
    max_length = json.loads(f.read())["max_length"]
f.close()

config = {
    "train_batch_size": 8,
    "valid_batch_size": 8,
    "weight_decay": 0.1,
    "shuffle_buffer": 1000,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "num_warmup_steps": 750,
    "gradient_accumulation_steps": 16,
    "max_train_steps": 50_000,
    "max_eval_steps": -1,
    "seq_length": max_length,
    "seed": 628,
    "save_checkpoint_steps": 50_000,
    "save_dir": "./models/JustSumAI",
    "model_name": model_name
}
args = Namespace(**config)

# ### Load dataset
BATCH_SIZE = args.train_batch_size

dataset = load_dataset(f"Dragonoverlord3000/JustSumAI_cleaned_gpt2_data", streaming=True)
print(dataset)

class SumDataset(Dataset):
    def __init__(self, dataset=dataset, split="train", max_length=args.seq_length, vocabulary=vocabulary):
        self.dataset = []
        for element in dataset[split]:
            self.dataset.append(
                json.loads(element["text"])
            )
        self.dataset_size = len(self.dataset)
        self.max_length = max_length
        self.vocabulary = vocabulary
        
    def __getitem__(self, idx):
        data_point = self.dataset[idx]
        # Make sure to pad output to max length
        data_point += [self.vocabulary.mask_index] * (self.max_length - len(data_point))
        data_point = torch.LongTensor(data_point)
        return data_point
    
    def __len__(self):
        return self.dataset_size

train_dataset = SumDataset()
eval_dataset = SumDataset(split="validation")

for i, data in enumerate(train_dataset):
    if i > 0: break

# Print sample data
print(data)
l = [vocabulary.getToken(idx.item()) for idx in data]
print(l)
eq = Equation([Token(t) for t in l[l.index("[SEP]")+1:] if t not in ["<MASK>", "<END>"]], notation="postfix")
print(eq.getMathmetaicalNotation())

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(train_dataloader, eval_dataloader)

for i,data in enumerate(train_dataloader):
    if i > 0:
        break
    test = data
    
print(test, test[0], BATCH_SIZE)

# ### Load model
model = AutoModelForCausalLM.from_pretrained(f"Dragonoverlord3000/{model_name}")
print(model)

print(f"GPT-2 Number of parameters: {model.num_parameters()/1_000_000:.2f}M")

# ### Define weight decay parameters

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n,p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
            
        return [{"params": params_with_wd, "weight_decay": args.weight_decay},
               {"params": params_without_wd, "weight_decay": 0.0}]


# ### Setting up logging for the training loop

def evaluate():
    model.eval()
    losses = []
    for step,batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps: break
    loss = torch.mean(torch.cat(losses))
    # Lower perplexity implies better performance
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float("inf"))
    return loss.item(), perplexity.item()


# # Training Loop
# Set seed for model
set_seed(args.seed)

# Accelerator
accelerator = Accelerator()
samples_per_step = accelerator.state.num_processes * args.train_batch_size
print(samples_per_step, accelerator.is_main_process)

# Clone model repository
if accelerator.is_main_process:
    hf_repo = Repository("../")


print(f"{accelerator.state}")

# Prepare the optimizer and learning rate scheduler
optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)
lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                            num_warmup_steps=args.num_warmup_steps, 
                            num_training_steps=args.max_train_steps)

accelerator.register_for_checkpointing(lr_scheduler)


def get_lr():
    return optimizer.param_groups[0]["lr"]


# Prepare everything  with our `accelerator` (order of args is not important)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader)

# ### Training Time

# Train model
model.train()
completed_steps = 0
for step, batch in enumerate(train_dataloader, start=1):
    loss = model(batch, labels=batch).loss
    loss /= args.gradient_accumulation_steps
    accelerator.backward(loss)
    if step % args.gradient_accumulation_steps == 0:
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1
        
    if step % args.save_checkpoint_steps == 0:
        eval_loss, perplexity = evaluate()
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            model.save_pretrained(f"models/{model_name}", 
                      push_to_hub=True, 
                      organization="Dragonoverlord3000")
        if completed_steps >= args.max_train_steps:
            break


