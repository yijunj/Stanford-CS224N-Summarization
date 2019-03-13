import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Set, Union
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class DocumentEncoder(nn.Module):
    def __init__(self, hidden_size, sent_hidden_size, doc_dropout_rate=0.3):
        """
        @param hidden_size (int): the hidden size of the sentence encoder. After concatenation, it becomes 2*hidden_size
        @param sent_hidden_size (int): hidden size of the doc-level encoder = size of document-level sentence encoding / 2
        @param doc_dropout_rate (float): Dropout probability
        """
        super(DocumentEncoder, self).__init__()
        self.doc_encoder = nn.GRU(2*hidden_size, sent_hidden_size, num_layers=1, bidirectional=True)
        self.doc_dropout = nn.Dropout(doc_dropout_rate)

    def forward(self, doc_padded: torch.Tensor, doc_lengths: List[int]) -> torch.Tensor:
        """
        !!!!!!!!!!!!!!!TODO: sort the doc_padded in order of longest to shortest sentence!!!!!!!!!!!!!!!!!
        @param doc_padded (Tensor): tensor of encoded sentence from the sentence-level encoder. Shape (doc_len, batch_size, sent_embed_size), where doc_len = maximum number of sentences in a document, sent_embed_size = 2*hidden_size. Note that these have already been sorted in order of longest to shortest sentence.
        @param doc_lengths (List[int]):  List of actual lengths for each of the documents in the batch
        @returns sent_enc_doc (Tensor): Tensor of hidden units with shape (batch_size, doc_len, sent_hidden_size*2)
        """
        X_packed = pack_padded_sequence(doc_padded, doc_lengths, batch_first = True)
        outputs, sent_enc_doc_hiddens = self.doc_encoder(X_packed) # now outputs has the shape (seq_len, batch, num_directions * hidden_size)
        outputs = pad_packed_sequence(outputs)[0].permute([1,0,2]) # now outputs has the shape (batch, seq_len, num_directions * hidden_size)
        sent_enc_doc = self.doc_dropout(outputs)

        return sent_enc_doc
