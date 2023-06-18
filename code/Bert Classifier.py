#!/usr/bin/env python
# coding: utf-8

# In[16]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from datasets import Dataset

from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch


# In[17]:
from datasets import disable_caching
disable_caching()


model_ckpt = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# # Load and analyse data

# In[18]:
# In[21]:


token2int = {token:i for i,token in enumerate(['Pi','Catalan','EulerGamma','Sqrt','Log'])}
num_labels = len(token2int)
print(token2int)


# In[22]:


def sum2tokens(mathematica_sum):
    """
    Example:
        '{{}, {-2, -2, -10, -8, -9, -8, -10, -5, -6, -4}}\n' ---> ['#', '/', '-2', '-2', '-10', '-8', '-9', '-8', '-10', '-5', '-6', '-4']

    """
    LHS, RHS = mathematica_sum.split("}, {")
    LHS, RHS = LHS.lstrip("{"), RHS.rstrip("\n").rstrip("}")
    if len(LHS) == 0:
        LHS = ["#"]
    else:
        LHS = LHS.split(", ")
    RHS = RHS.split(", ")    
    return LHS + ["/"] + RHS
sum2tokens('{{}, {-2, -2, -10, -8, -9, -8, -10, -5, -6, -4}}\n')


# In[23]:
dataset_encoded = load_dataset("Dragonoverlord3000/sum_classifier_dataset", streaming=True)

# # Model

# In[ ]:


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# In[ ]:


model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)
print(model)


# In[ ]:


# The training arguments - note the `push_to_hub`
batch_size = 32
logging_steps = batch_size
model_name = "sum-classifier"
training_args = TrainingArguments(output_dir=model_name,
                                 num_train_epochs=10,
                                 learning_rate=2e-5,
                                 per_device_train_batch_size=batch_size,
                                 per_gpu_eval_batch_size=batch_size,
                                 weight_decay=0.01,
                                 evaluation_strategy="epoch",
                                 disable_tqdm=False,
                                 logging_steps=logging_steps,
                                 push_to_hub=True,
                                 log_level="error")


# In[ ]:


# Define the trainer
trainer = Trainer(model=model, args=training_args,
                 compute_metrics=compute_metrics,
                 train_dataset=dataset_encoded,
                 eval_dataset=dataset_encoded,
                 tokenizer=tokenizer)


# In[ ]:


trainer.push_to_hub(commit_message="Training Complete")


# In[ ]:





# In[ ]:





# In[ ]:



