from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_scheduler
from model.equation_interpreter import Equation
from torch.utils.data import IterableDataset, DataLoader
from argparse import Namespace
from typing import List
import linecache
import json
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from accelerate import Accelerator
from huggingface_hub import Repository
from torch.optim import AdamW
from tqdm import tqdm

# Domestic imports
from model.tokens import TOKEN_TYPE_ANSWERS, TOKEN_TYPE_EXPRESSIONS, Token
from model.equation_interpreter import Equation, EquationLexer
from model.vocabulary import Vocabulary

model_ckpt = "JustSumAI2"
org = "Dragonoverlord3000"
model_id = f"{org}/{model_ckpt}"


def expression2tokens(expression):
    """
    Example:
        '{{}, {-2, -2, -10, -8, -9, -8, -10, -5, -6, -4}}\n' ---> ['#', '/', '-2', '-2', '-10', '-8', '-9', '-8', '-10', '-5', '-6', '-4']

    """
    LHS, RHS = expression.split("}, {")
    LHS, RHS = LHS.lstrip("{"), RHS.rstrip("\n").rstrip("}")
    if len(LHS) == 0:
        LHS = ["#"]
    else:
        LHS = LHS.split(", ")
    RHS = RHS.split(", ")    
    return LHS + ["/"] + RHS

def answer2tokens(answer):
    """
    Example:
        '(571057069 - 57859200*Pi^2)/1365590016000\n' ---> [Token(TT_INTEGER), Token(TT_INTEGER), Token(TT_PI), Token(TT_INTEGER), Token(TT_POW), Token(TT_MULTIPLY), Token(TT_MINUS), Token(TT_INTEGER), Token(TT_DIVIDE)]
    """
    equation = Equation.makeEquationFromString(answer)
    equation.convertToPostfix()
    if equation.notation != "postfix": return None
    return [token.t_type for token in equation.tokenized_equation]

vocabulary = Vocabulary.construct_from_list(TOKEN_TYPE_EXPRESSIONS + TOKEN_TYPE_ANSWERS)
base_vocab = (list(vocabulary.token2index.keys())[:-5] + ["[SEP]"])
base_vocab.remove("TT_RATIONAL")
base_vocab.remove("TT_VARIABLE")
print(base_vocab, len(base_vocab))

def dummy_iterator():
    yield []

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = tokenizer.train_new_from_iterator(
    dummy_iterator(),
    vocab_size=257 # Minimum allowed size
) # Just a way to initialize a new tokenizer of the same 'type' meant for GPT2
tokenizer.add_tokens(base_vocab)
print(tokenizer)

# Sanity check
assert len([token for token in tokenizer.get_vocab() if token in base_vocab]) == len(base_vocab)

tokenizer.save_pretrained(model_id, push_to_hub=True)

config = {
    "answers_dir": "./data/answers-1000.txt",
    "expressions_dir": "./data/expressions-1000.txt",
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
    "seq_length": 1024, # We use a buffer to fill out the seq_length
    "seed": 628,
    "save_checkpoint_steps": 50_000,
    "save_dir": "./models/JustSumAI",
    "model_name": model_ckpt,
    "num_epochs": 100
}
args = Namespace(**config)

class SumDataset(IterableDataset):
    def __init__(self, tokenizer, answers_dir=args.answers_dir, expressions_dir=args.expressions_dir, seq_length=args.seq_length) -> None:
        """
        Args:
            answers_dir (str): directory to file containing string equations e.g. '(571057069 - 57859200*Pi^2)/1365590016000\n'
            expressions_dir (str): directory to file containing expressions e.g. '{{}, {-2, -2, -10, -8, -9, -8, -10, -5, -6, -4}}\n'
            
        Note: Initializatio of the dataset might take a few seconds
        """
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.input_characters = seq_length * 3.6 * 1024
        
        self.expressions_dir = expressions_dir
        self.answers_dir = answers_dir
        self.seq_length = seq_length
        with open(expressions_dir, "r") as fe:
            self.dataset_size = len(fe.read().split("\n"))
        fe.close()
        self.dataset = []
        for line_num in range(1,self.dataset_size):
            LHS, RHS = linecache.getline(expressions_dir, line_num), linecache.getline(answers_dir, line_num)
            try:
                LHS = expression2tokens(LHS)
                RHS = answer2tokens(RHS)
                if LHS and RHS:
                    self.dataset.append(" ".join(LHS + ["[SEP]"] + RHS))
            except:
                pass
    
    def __iter__(self) -> torch.Tensor:
        iterator = iter(self.dataset)
        # This is an infinite generator over the dataset
        while True:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    m=f"Buffer full: {buffer_len}>={self.input_characters:.0f}"
                    print(m)
                    break
                try:
                    m=f"Fill buffer: {buffer_len}<{self.input_characters:.0f}"
                    print(m)
                    buffer.append(next(iterator))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    iterator = iter(self.dataset)

            all_token_ids = []
            tokenized_inputs = self.tokenizer(buffer, truncation=False)
            for tokenized_input in tokenized_inputs['input_ids']:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)
        
    
    def __len__(self) -> int:
        return self.dataset_size
    
    
constant_length_dataset = SumDataset(tokenizer)
dataloader = DataLoader(constant_length_dataset, batch_size=args.train_batch_size)

model = AutoModelForCausalLM.from_pretrained(f"Dragonoverlord3000/{model_ckpt}")
print(model)

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n,p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
            
        return [{"params": params_with_wd, "weight_decay": args.weight_decay},
               {"params": params_without_wd, "weight_decay": 0.0}]
    
def evaluate():
    model.eval()
    losses = []
    for step,batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(batch["data"], labels=batch["labels"])
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
    
# Accelerator
accelerator = Accelerator()
samples_per_step = accelerator.state.num_processes * args.train_batch_size
print(samples_per_step, accelerator.is_main_process)
    
# Clone model repository
if accelerator.is_main_process:
    hf_repo = Repository("../")
print(accelerator.state)

# Prepare the optimizer and learning rate scheduler
optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)
lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                            num_warmup_steps=args.num_warmup_steps, 
                            num_training_steps=args.max_train_steps)

accelerator.register_for_checkpointing(lr_scheduler)

def get_lr():
    return optimizer.param_groups[0]["lr"]

# Prepare everything  with our `accelerator` (order of args is not important)
model, optimizer, dataloader, dataloader = accelerator.prepare(
    model, optimizer, dataloader, dataloader)

# Train model
model.train()
completed_steps = 0
for epoch in tqdm(range(args.num_epochs)):
    for step, batch in tqdm(enumerate(dataloader, start=1)):
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
            if accelerator.is_main_process:
                model.save_pretrained(f"models/{model_ckpt}", 
                          push_to_hub=True, 
                          organization="Dragonoverlord3000")

            model.train()
            if completed_steps >= args.max_train_steps:
                break
                
model.save_pretrained(f"models/{model_ckpt}", 
                          push_to_hub=True, 
                          organization="Dragonoverlord3000")





