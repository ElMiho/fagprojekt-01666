# Imports
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tqdm import tqdm
import numpy as np
import linecache
import argparse
import json
import time
import os

from model.equation_interpreter import Equation
from model.vocabulary import Vocabulary
from model.vocabulary import vocabulary_answers as target_vocabulary
from model.vocabulary import vocabulary_expressions as source_vocabulary

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


# Setup
device = "cuda" if config["use_cuda"] and torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


###########
# DATASET #
###########
class SumDataset(Dataset):
    def __init__(self, inputs_file:str=config["inputs_file"], targets_file:str=config["targets_file"],
                 input_vocab:Vocabulary=source_vocabulary, target_vocab:Vocabulary=target_vocabulary) -> None:
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
            self.max_seq_length_target = max(self.max_seq_length_target, len(json.loads(linecache.getline(self.targets_file, index))))
    
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
        input_idx_list.extend([self.input_vocab.mask_index] * (self.max_seq_length_input - len(input_idx_list)))
        
        target_idx_list_x = target_idx_list[:-1]
        target_idx_list_y = target_idx_list[1:]
        target_idx_list_x.extend([self.target_vocab.mask_index] * (self.max_seq_length_target - len(target_idx_list_x)))
        target_idx_list_y.extend([self.target_vocab.mask_index] * (self.max_seq_length_target - len(target_idx_list_y)))

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

def generate_nmt_batches(dataset):
    """A generator function which wraps the PyTorch DataLoader. The NMT Version """
    dataloader = DataLoader(dataset, 
        batch_size=config["batch_size"],    # samples data into collections
        shuffle=config["shuffle"],          # shuffles the indices
        drop_last=config["drop_last"]       # drop the last batch if len(data) does not divide batch_size
    )

    for batch_index, data_dict in enumerate(dataloader):
        lengths = data_dict['input_lengths'].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()
        
        out_data_dict = {}
        for name in data_dict:
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield batch_index, out_data_dict 

if args.verbose:
    print(f"Initializing dataset...")
dataset = SumDataset(
    inputs_file=config["inputs_file"],
    targets_file=config["targets_file"]
)
if args.verbose:
    print(f"Dataset `{dataset}` initialized!")


# 
# Structure and code is loosely modeled based on the following jupyter notebook: https://github.com/delip/PyTorchNLPBook/blob/master/chapters/chapter_8/8_5_NMT/8_5_NMT_No_Sampling.ipynb
# Article describing pytorch's built in RNN module (i.e. sizes and such): https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-rnn-cb6ebc594677
# 

#########
# MODEL #
#########
class Encoder(nn.Module):
    def __init__(self, num_embeddings:int, embedding_size:int, rnn_hidden_size:int, padding_idx=source_vocabulary.mask_index) -> None:
        """
        Args:
            num_embeddings: number of embeddings is the input (expressions) vocabulary size
            embedding_size: size of the embedding vectors
            rnn_hidden_size: size of the RNN hidden state vectors
        """
        super().__init__()

        # Embed the input (source) sequence
        self.source_embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=padding_idx)
        # Bidirectional Gated Recurrent Unit
        self.birnn = nn.GRU(embedding_size, rnn_hidden_size, 
                            num_layers=1,
                            bidirectional=True, batch_first=True)

    def forward(self, x_source: torch.Tensor, x_lengths: torch.Tensor):
        """The forward pass of the model
        
        Args:
            x_source: the padded input (expressions) data index tensor
            x_lengths: explicit lengths of non-padded sequences
        
        Returns:
            a tuple:  (x_unpacked, x_birnn_h)
                x_unpacked.shape = (batch size, sequence length, 2*rnn_hidden_size)
                x_birnn_h.shape = (batch size, sequence length, 2*rnn_hidden_size)
        """
        x_embedded = self.source_embedding(x_source)
        # create PackedSequence; x_packed.data.shape=(number_items, embedding_size)
        # PackedSequence is just a CUDA optimized representation of our embedded input
        x_packed = pack_padded_sequence(x_embedded, 
                                        x_lengths.detach().cpu().numpy(),
                                        batch_first=True)
        
        # Note, the factor 2 is due to us using a bidirectional RNN
        ## x_birnn_out is the collection of outputs from all time steps in the RNN, shape = (sequence length, batch size, 2*rnn_hidden_size)
        ## x_birnn_h is the final hidden state outputted by the RNN, shape = (2*num_rnn_layers, batch size, rnn_hidden_size)
        x_birnn_out, x_birnn_h = self.birnn(x_packed)
        # permute to (batch_size, num_rnn, feature_size)
        x_birnn_h = x_birnn_h.permute(1,0,2)

        # flatten features
        x_birnn_h = x_birnn_h.contiguous().view(x_birnn_h.size(0), -1)

        # The reverse of `pack_padded_sequence`, will map to shape (batch size, seq length, feature size)
        x_unpacked, _ = pad_packed_sequence(x_birnn_out, batch_first=True)

        return x_unpacked, x_birnn_h

def verbose_attention(encoder_state_vectors, query_vector):
    """A descriptive version of the neural attention mechanism 
    
    Args:
        encoder_state_vectors (torch.Tensor): 3dim tensor from bi-GRU in encoder
            encoder_state_vectors is x_unpacked from the encoder outputs
        query_vector (torch.Tensor): hidden state in decoder GRU
            query_vector is a hidden state vector from a single time step of the decoder
    Returns:
        context_vector, vector_probabilities, vector_scores
    """
    # Get size of encoder states
    # Read for understanding attention: https://lilianweng.github.io/posts/2018-06-24-attention/
    batch_size, num_vectors, vector_size = encoder_state_vectors.size()
    vector_scores = torch.sum(encoder_state_vectors * query_vector.view(batch_size, 1, vector_size), 
                              dim=2)
    vector_probabilities = torch.softmax(vector_scores, dim=1)
    weighted_vectors = encoder_state_vectors * vector_probabilities.view(batch_size, num_vectors, 1)
    context_vectors = torch.sum(weighted_vectors, dim=1)

    return context_vectors, vector_probabilities, vector_scores

class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, 
                 bos_index, padding_idx=target_vocabulary.mask_index) -> None:
        """
        Args:
            num_embeddings: the number of words in the target vocabulary
            embedding_size: hyperparameter, the length of the embedding vector
            rnn_hidden_size: size of a rnn hidden state
            bos_index: the begin of sentence token in the target vocabulary
        """
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size

        # The embedding of the output space language
        self.target_embedding = nn.Embedding(num_embeddings=num_embeddings,
                                             embedding_dim=embedding_size,
                                             padding_idx=padding_idx)
        # Gated recurrent unit cell, single step/building block of a GRU-based RNN
        ## Note: in our implementation, the input to gru_cell is a single embedded word (the 'new' word) 
        ## concatenated with the current context vector
        self.gru_cell = nn.GRUCell(input_size=embedding_size+rnn_hidden_size,
                                   hidden_size=rnn_hidden_size)
        # Linear map on the initial hidden state (from the encoder)
        self.hidden_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        # Linear classifier map on the context vector combined with the current hidden vector
        self.classifier = nn.Linear(2 * rnn_hidden_size, num_embeddings)
        # Dropout layer for regulariation
        self.dropout = nn.Dropout(0.3)
        # The begin of sentence index for the target sequence
        self.bos_index = bos_index
        # See: https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277
        ## Essentially: temperatures > 1 lowers confidence in prediction and temperatures < 1 increases confidence
        self._sampling_temperature = 3

    def forward(self, encoder_state, initial_hidden_state, target_sequence, sample_probability=0):
        """The forward pass of the model
        
        Args:
            encoder_state: the output of the encoder, 
                i.e. the output from all time steps
            initial_hidden_state: the final hidden state of the encoder
                i.e. the output from the last time step in each layer
            target_sequence: the target text data tensor
                is used in training only
            sample_probability: probability of using output from model as next input
                as opposed to using the target sequence directly

        Returns:
            output_vectors: prediction vectors at each output step
        """
        if target_sequence is None:
            sample_probability = 1
            output_sequence_size = dataset.max_seq_length_target
        else:
            # Assumes batch first
            ## Permutes (batch, sequence) --> (sequence, batch)
            target_sequence = target_sequence.permute(1,0)
            output_sequence_size = target_sequence.size(0)

        batch_size = encoder_state.size(0)

        # use the provided encoder hidden state as the initial hidden state
        h_t = self.hidden_map(initial_hidden_state)

        # Initialize context vectors to zero vector
        context_vectors = torch.zeros(batch_size, self.rnn_hidden_size).to(device)
        # Initialize first y_t word as BOS
        y_t_index = torch.ones(batch_size, dtype=torch.int64) * self.bos_index

        # Use same device as the encoder state
        h_t = h_t.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)

        # Keep track of decoder history
        output_vectors = []
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()

        for i in range(output_sequence_size):
            # Whether to use self-generated sample or use target directly
            use_sample = np.random.random() < sample_probability
            
            if not use_sample:
                y_t_index = target_sequence[i]

            # Step 1: Embed word and concat with previous context
            y_input_vector = self.target_embedding(y_t_index)
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)

            # Step 2: Make a GRU step, getting a new hidden vector
            h_t = self.gru_cell(rnn_input, h_t)
            self._cached_ht.append(h_t.cpu().detach().numpy())
            
            # Step 3: Use the current hidden to attend to the encoder state
            context_vectors, p_attn, _ = verbose_attention(encoder_state_vectors=encoder_state, 
                                                           query_vector=h_t)
            
            # auxillary: cache the attention probabilities for visualization
            self._cached_p_attn.append(p_attn.cpu().detach().numpy())
            
            # Step 4: Use the current hidden and context vectors to make a prediction to the next word
            prediction_vector = torch.cat((context_vectors, h_t), dim=1)
            score_for_y_t_index = self.classifier(
                self.dropout(prediction_vector)
            )

            if use_sample:
                p_y_t_index = torch.softmax(score_for_y_t_index * self._sampling_temperature, dim=1)
                # Take the maximum likely one
                # # _, y_t_index = torch.max(p_y_t_index, 1)
                # Take probabilistic sample
                y_t_index = torch.multinomial(p_y_t_index, 1).squeeze()
            
            # auxillary: collect the prediction scores
            output_vectors.append(score_for_y_t_index)

            # Break if inference and end predicted
            if sample_probability == 1 and (((not list(y_t_index.shape)) and y_t_index.item() == target_vocabulary.end_seq_index) or (list(y_t_index.shape) and all(v == target_vocabulary.end_seq_index for v in y_t_index))):
                break
            
        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)
        
        return output_vectors
    
class Model(nn.Module):
    def __init__(self, source_vocab_size, source_embedding_size, target_vocab_size,
                 target_embedding_size, encoding_size, target_bos_index) -> None:
        """
        Args:
            source_vocab_size: number of unique words in source language
            source_embedding_size: size of the source embedding vectors
            target_vocab_size: number of unique words in source language
            target_embedding_size: size of the source embedding vectors
            encoding_size: the size of the encoder RNN
                i.e. rnn_hidden_size of the encoder
        """
        super().__init__()
        self.encoder = Encoder(num_embeddings=source_vocab_size, embedding_size=source_embedding_size,
                               rnn_hidden_size=encoding_size)
        # Note the *2 is due to the encoder using a BIGRU
        self.decoder = Decoder(num_embeddings=target_vocab_size, embedding_size=target_embedding_size,
                               rnn_hidden_size=2*encoding_size, bos_index=target_bos_index)
        
    def forward(self, source, source_lengths, target_sequence):
        """The forward pass of the model
        
        Args:
            source: the source text data tensor
                source.shape = (batch_size, max_source_length)
            source_lengths: the length of the sequences in source
            target_sequence: the target text data tensor
        
        Returns:
            decoded_states: prediction vectors at each output step
        """
        encoder_state, final_hidden_states = self.encoder(source, source_lengths)
        decoded_states = self.decoder(encoder_state=encoder_state, 
                                      initial_hidden_state=final_hidden_states, 
                                      target_sequence=target_sequence)
        return decoded_states


################
# HELPER STUFF #
################

# For recording the training history
train_state = {
        "stop_early": False,
        "early_stopping_step": 0,
        "early_stopping_best_val": 1e+8,
        "learning_rate": config["learning_rate"],
        "epoch_index": 0,
        "train_loss": [],
        "train_acc": [],
        "model_filename": config["model_filename"]
    }

def normalize_sizes(y_pred, y_true):
    """Normalizes tensor sizes
    
    Args:
        y_pred: the output of the model
            If 3D tensor, reshape to 2D tensor (matrix)
        y_true: the target predictions
            If 2D tensor (matrix), reshape to 1D tensor (vector)
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true

def compute_accuracy(y_pred, y_true, mask_index):
    # Make sure y_pred has 2D shape and y_true a 1D shape
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    # torch.tensor.max(dim=x) returns a tensor of maximum values and their corresponding indices
    _, y_pred_indices = y_pred.max(dim=1)
    
    # Find every non-mask index that was correctly predicted
    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()
    
    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100

####################
# INITIALIZE MODEL #
####################

# The model
model = Model(
    source_vocab_size=len(source_vocabulary), source_embedding_size=config["embedding_size"],
    target_vocab_size=len(target_vocabulary), target_embedding_size=config["embedding_size"],
    encoding_size=config["rnn_hidden_size"], target_bos_index=source_vocabulary.begin_seq_index
)
model = model.to(device)

if args.verbose:
    print(f"Using model `{config['model_filename']}` with architecture:")
    print(model)

# Optimizer and training scheduler
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                           mode='min', factor=0.5,
                                           patience=1)


mask_index = target_vocabulary.mask_index
cross_entropy = nn.CrossEntropyLoss(ignore_index=mask_index)

# Get loss of prediction
def sequence_loss(y_pred, y_true, mask_index=target_vocabulary.mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return cross_entropy(y_pred, y_true)

#################
# TRAINING LOOP #
#################

## Note: tqdm just adds a progress bar to the training loop

epoch_iterator = tqdm(range(config["num_epochs"]), desc=f"Running loss: ---, Running acc: ---")
for epoch in epoch_iterator:
    train_state["epoch_index"] = epoch

    running_loss = 0
    running_acc = 0
    # Makes sure dropout is used
    dataloader = generate_nmt_batches(dataset)
    model.train()

    for batch_index, batch_dict in dataloader:
        # 1. zero the gradients
        optimizer.zero_grad()

        # 2. predict the output
        y_pred = model(
            batch_dict["input"],
            batch_dict["input_lengths"],
            batch_dict["target_x"]
        )

        # 3. compute the loss
        loss = sequence_loss(y_pred, batch_dict["target_y"], mask_index)

        # 4. use loss to calculate gradient
        loss.backward()

        # 5. optimizer
        optimizer.step()
        
        # Calculate the running loss and running accuracy
        running_loss += (loss.item() - running_loss) / (batch_index + 1)
        acc_t = compute_accuracy(y_pred, batch_dict["target_y"], mask_index)
        running_acc = (acc_t - running_acc) / (batch_index + 1)

    train_state["train_loss"].append(running_loss)
    train_state["train_acc"].append(running_acc)

    # Display running loss + save model
    epoch_iterator.set_description(f"Running loss: {round(running_loss, 4)}, Running acc: {round(running_acc, 4)}")
    torch.save(model.state_dict(), train_state['model_filename'])


#############
# INFERENCE #
#############
model.eval()

def to_indices(scores):
    _, indices = torch.max(scores, dim=1)
    return indices

def sentence_from_indices(indices, vocab, strict=True):
    out = []
    for index in indices:
        index = index.item()
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            return " ".join(out)
        else:
            out.append(vocab.getToken(index))
    return " ".join(out)

# n^2 test
test_expression = ["#", "/", "0", "0"]
test_tensor = torch.tensor([
    source_vocabulary.vectorize(test_expression) for _ in range(config["batch_size"])
], dtype=torch.int32)
test_pred = model(
    test_tensor,
    torch.LongTensor([len(test_tensor[0]) for _ in range(len(test_tensor))]),
    target_sequence=None
)

print(f"Test expression: {test_expression}")
print(f"Predicted shape: {test_pred.shape}")
print(f"Predicted value: {sentence_from_indices(to_indices(test_pred[0]), target_vocabulary)}")














