import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Set, Union
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from sent_encoder import SentenceEncoder
from doc_encoder import DocumentEncoder

class NeuSum(nn.Module):
    def __init__(self, word_embed_size, hidden_size, sent_hidden_size, vocab, sent_dropout_rate=0.3, doc_dropout_rate=0.2):
        """
        @param word_embed_size (int): embedding size of a word
        @param hidden_size (int): size of the output of a forward/backard GRU
        @param sent_hidden_size (int): hidden size of the doc-level encoder = size of document-level sentence encoding / 2
        @param vocab (Vocab): Vocabulary object
        @param sent_dropout_rate (float): Dropout probability for setence-level encoder
        @param doc_dropout_rate (float): Dropout probability for document-level encoder
        """
        super(NeuSum, self).__init__()
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.vocab = vocab
        self.sent_dropout_rate = sent_dropout_rate
        self.doc_dropout_rate =  doc_dropout_rate

        self.sentence_encoder = SentenceEncoder(word_embed_size, hidden_size, vocab)
        self.document_encoder = DocumentEncoder(hidden_size, sent_hidden_size)

    def forward(self, input):
        """
        @param input (List[List[List[str]]]): input documents, every word is a separate string
        @param
        """
        # turn input to a Tensor: idx of words and padded on both sentence and document levels
        input_idx = self.vocab.to_input_tensor(input, device=torch.device('cpu')) # shape: (batch_size, doc_len, src_len)
        batch_size, doc_len, src_len = input_idx.size()

        # TODO: flatten the input_idx to source_padded (src_len, batch_size*doc_len)
        input_idx = input_idx.permute(2, 0, 1) # (src_len, batch_size, doc_len)
        source_padded = input_idx.contiguous().view(src_len, -1) # shape: (src_len, batch_size*doc_len)
        # later just take every doc_len

        # sent_enc has shape: (batch_size*doc_len, hidden_size*2)
        sent_enc = self.sentence_encoder(source_padded, [src_len]*source_padded.size()[1])

        # TODO: transform sent_enc (batch_size*doc_len, hidden_size*2) to doc_padded (doc_len, batch_size, hidden_size*2)
        doc_padded = sent_enc.contiguous().view(batch_size, doc_len, -1).permute(1, 0, 2)# shape: (doc_len, batch_size, sent_embed_size)

        # document-level encoder
        sent_enc_doc = self.document_encoder(doc_padded, [doc_len]*doc_padded.size()[1]) # shape: (batch_size, doc_len, sent_hidden_size*2)

        return sent_enc_doc
