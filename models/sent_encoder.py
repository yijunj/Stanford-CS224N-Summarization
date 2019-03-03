import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Set, Union
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings

class SentenceEncoder(nn.Module):
    def __init__(self, word_embed_size, hidden_size, sent_dropout_rate=0.3):
        """
        @param word_emb_size (int)
        @param hidden_size (int): size of the output of a forward/backard GRU
        @param sent_dropout_rate (float): Dropout probability
        """
        super(SentenceEncoder, self).__init__()
#         self.vocab = vocab
        self.model_embeddings = ModelEmbeddings(word_embed_size, vocab)
        self.sent_encoder = nn.GRU(word_embed_size, hidden_size, num_layers=1, bidirectional=True)
        self.sent_dropout = nn.Dropout(sent_dropout_rate)

    def forward(self, source_padded: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """
        !!!!!!!!!!!!!!!TODO: sort the source_padded in order of longest to shortest sentence!!!!!!!!!!!!!!!!!
        @param source_padded (Tensor): indices of words, shape (src_len, b). Note that these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns sent_enc (Tensor): Tensor of encoded sentences with shape (batch_size, hidden_size*2)
        """
        # source_idx = self.vocab(source_padded) # word idx
        X = self.model_embeddings.source(source_padded) # shape: (src_len, b, word_embed_size)
        X_packed = pack_padded_sequence(X, source_lengths)
        outputs, sent_enc_hiddens = self.sent_encoder(X_packed)
        #sent_enc_hiddens has shape (num_layers * num_directions, batch, hidden_size), outputs has shape (seq_len, batch, num_directions * hidden_size):
        # print(sent_enc_hiddens.size())
        sent_enc = torch.cat( (sent_enc_hiddens[0,:], sent_enc_hiddens[1,:]), dim=1 ) #concatenate the hidden states from forward and backward GRUs
        # print(sent_enc.size())

        # outputs = pad_packed_sequence(outputs)[0].permute([1,0,2])
        sent_enc = self.sent_dropout(sent_enc) #shape: (batch, hidden_size*2)
        # outputs = self.sent_dropout(outputs)
        # print(outputs.size())

        return sent_enc
