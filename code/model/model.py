import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.vocabulary import vocabulary_answers as target_vocabulary
from model.vocabulary import vocabulary_expressions as source_vocabulary

import numpy as np

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
    
    Read for understanding attention: https://lilianweng.github.io/posts/2018-06-24-attention/

    Args:
        encoder_state_vectors (torch.Tensor): 3dim tensor from bi-GRU in encoder
            encoder_state_vectors is x_unpacked from the encoder outputs
        query_vector (torch.Tensor): hidden state in decoder GRU
            query_vector is a hidden state vector from a single time step of the decoder
    Returns:
        context_vector, vector_probabilities, vector_scores
    """
    # Get size of encoder states
    batch_size, num_vectors, vector_size = encoder_state_vectors.size()
    vector_scores = torch.sum(encoder_state_vectors * query_vector.view(batch_size, 1, vector_size), 
                              dim=2)
    vector_probabilities = torch.softmax(vector_scores, dim=1)
    weighted_vectors = encoder_state_vectors * vector_probabilities.view(batch_size, num_vectors, 1)
    context_vectors = torch.sum(weighted_vectors, dim=1)

    return context_vectors, vector_probabilities, vector_scores

class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, 
                 bos_index, max_seq_length, padding_idx=target_vocabulary.mask_index) -> None:
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
        self.dropout = nn.Dropout(0.3, inplace=True)
        # Maximum target sequence length
        self.max_seq_length = max_seq_length
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
            output_sequence_size = self.max_seq_length
        else:
            # Assumes batch first
            ## Permutes (batch, sequence) --> (sequence, batch)
            target_sequence = target_sequence.permute(1,0)
            output_sequence_size = target_sequence.size(0)

        batch_size = encoder_state.size(0)

        # use the provided encoder hidden state as the initial hidden state
        h_t = self.hidden_map(initial_hidden_state)

        # Initialize context vectors to zero vector
        context_vectors = torch.zeros(batch_size, self.rnn_hidden_size).to(encoder_state.device)
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
                 target_embedding_size, encoding_size, target_bos_index, max_seq_length) -> None:
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
                               rnn_hidden_size=2*encoding_size, max_seq_length=max_seq_length, 
                               bos_index=target_bos_index)
        
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

