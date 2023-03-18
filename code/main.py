# Imports
import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import linecache
import argparse
import json
import os

from model.equation_interpreter import Equation
from model.vocabulary import vocabulary_answers, vocabulary_expressions, Vocabulary

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
                 input_vocab:Vocabulary=vocabulary_answers, target_vocab:Vocabulary=vocabulary_expressions) -> None:
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

if args.verbose:
    print(f"Initializing dataset...")
dataset = SumDataset(
    inputs_file=config["inputs_file"],
    targets_file=config["targets_file"]
)
dataloader = DataLoader(dataset, 
    batch_size=config["batch_size"],    # samples data into collections
    shuffle=config["shuffle"],          # shuffles the indices
    drop_last=config["drop_last"]       # drop the last batch if len(data) does not divide batch_size
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
class Encoder:
    def __init__(self, num_embeddings:int, embedding_size:int, rnn_hidden_size:int, padding_idx=dataset.input_vocab.mask_index) -> None:
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
        x_packed = pack_padded_sequence(x_embedded, x_lengths.detach().cpu().numpy(),
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

class Decoder:
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, 
                 bos_index, padding_idx=dataset.target_vocab.mask_index) -> None:
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
        # The begin of sentence index for the target sequence
        self.bos_index = bos_index

    def forward(self, encoder_state, initial_hidden_state, target_sequence):
        """The forward pass of the model
        
        Args:
            encoder_state: the output of the encoder, 
                i.e. the output from all time steps
            initial_hidden_state: the final hidden state of the encoder
                i.e. the output from the last time step in each layer
            target_sequence: the target text data tensor

        Returns:
            output_vectors: prediction vectors at each output step
        """
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
            y_t_index = target_sequence[i]













