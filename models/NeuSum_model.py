import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Set, Union
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from models.sent_encoder import SentenceEncoder
from models.doc_encoder import DocumentEncoder

class NeuSum(nn.Module):
    def __init__(self, word_embed_size, hidden_size, sent_hidden_size, extract_hidden_size, vocab, device, sent_dropout_rate=0.3, doc_dropout_rate=0.2, max_t_step = 3, attn_size = 512):
        """
        @param word_embed_size (int): embedding size of a word
        @param hidden_size (int): size of the output of a forward/backard GRU
        @param sent_hidden_size (int): hidden size of the doc-level encoder = size of document-level sentence encoding / 2
        @param extract_hidden_size: hidden size of the extraction GRU
        @param vocab (Vocab): Vocabulary object
        @param sent_dropout_rate (float): Dropout probability for setence-level encoder
        @param doc_dropout_rate (float): Dropout probability for document-level encoder
        @param max_t_step: maximum number of time steps for setence selection
        @attn_size: output size of Wq*ht and Wd*si in the scorer
        """
        super(NeuSum, self).__init__()
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.vocab = vocab
        self.sent_dropout_rate = sent_dropout_rate
        self.doc_dropout_rate =  doc_dropout_rate
        self.extract_hidden_size = extract_hidden_size
        self.max_t_step = max_t_step
        self.device = device

        self.sentence_encoder = SentenceEncoder(word_embed_size, hidden_size, vocab)
        self.document_encoder = DocumentEncoder(hidden_size, sent_hidden_size)

        self.h0_linear = nn.Linear(sent_hidden_size, extract_hidden_size)
        self.ht_gru = nn.GRUCell(sent_hidden_size*2, extract_hidden_size)
        self.score_linear_q = nn.Linear(extract_hidden_size, attn_size)
        self.score_linear_d = nn.Linear(sent_hidden_size*2, attn_size)
        self.score_linear_s = nn.Linear(attn_size, 1)


    def encode(self, input):
        """
        @param input (List[List[List[str]]]): input documents, every word is a separate string
        @returns sent_enc_doc (Tensor): encoded sentences on doc-level, shape (batch_size, doc_len, sent_hidden_size*2)
        """
        # turn input to a Tensor: idx of words and padded on both sentence and document levels
        input_idx = self.vocab.to_input_tensor(input, device=torch.device('cpu')) # shape: (batch_size, doc_len, src_len)
        self.batch_size, self.doc_len, self.src_len = input_idx.size()

        # flatten the input_idx to source_padded (src_len, batch_size*doc_len) ....
        input_idx = input_idx.permute(0,2,1) # (2,0,1) = (src_len, batch_size, doc_len)
        source_padded = input_idx.contiguous().view(self.src_len, -1) # shape: (src_len, batch_size*doc_len)
        # later just take every doc_len
        #print(source_padded.shape)

        # sent_enc has shape: (batch_size*doc_len, hidden_size*2)
        sent_enc = self.sentence_encoder(source_padded, [self.src_len]*source_padded.size()[1])

        # transform sent_enc (batch_size*doc_len, hidden_size*2) to doc_padded (doc_len, batch_size, hidden_size*2)
        doc_padded = sent_enc.contiguous().view(self.batch_size, self.doc_len, -1).permute(1, 0, 2)# shape: (doc_len, batch_size, sent_embed_size)

        # document-level encoder
        sent_enc_doc = self.document_encoder(doc_padded, [self.doc_len]*doc_padded.size()[1]) # shape: (batch_size, doc_len, sent_hidden_size*2)

        return sent_enc_doc

    def create_score_mask(self, sents_selected):
        """
        @param sents_selected (list of Tensors): sentences selected so far. Tensor shape: (batch_size,)
        @returns score_mask (Tensor): shape (batch_size, doc_len), with 1 for selected sentences and 0 for unselected sentences
        """
        score_mask = torch.zeros(self.batch_size, self.doc_len).byte()
        for sent_selected_t in sents_selected:
            for doc_num, sent_idx in enumerate(sent_selected_t):
                score_mask[doc_num, sent_idx] = 1

        return score_mask.to(self.device)

    def score_selection(self, sent_enc_doc):
        """
        @param sent_enc_doc (Tensor): document-level encodings. Shape: (batch_size, doc_len, sent_hidden_size*2). Output from the encoder
        @returns sent_scores (Tensor): sentence scores of every time step. Shape (batch_size, T, doc_len) where T=3 is the total number of time steps (max_t_step)
        @returns sent_selected (Tensor): The indices of selected three sentences of each document. Shape (batch_size, 3)
        """
        # sent_hidden_size = sent_enc_doc.view()[2] / 2
        last_back_vec = sent_enc_doc[:,0,self.sent_hidden_size:self.sent_hidden_size*2] #shape (batch_size, sent_hidden_size)
        last_back_vec_flat = last_back_vec.repeat(1, self.doc_len).contiguous().view(-1,self.sent_hidden_size) #shape (batch_size*doc_len, sent_hidden_size)

        # initialization
        h_prev = torch.tanh(self.h0_linear(last_back_vec_flat)) # shape (batch_size*doc_len, extract_hidden_size)
        sents_selected = [] # 'S0' in pdf
        sents_scores = []
        sent_selected_t = None
        s_prev = torch.zeros(self.batch_size*self.doc_len, self.sent_hidden_size*2) # 's0' in pdf
        sent_enc_doc_flat = sent_enc_doc.contiguous().view(-1, self.sent_hidden_size*2) # shape (batch_size*doc_len, extract_hidden_size)

        for t in range(self.max_t_step):
            h_next = self.ht_gru(s_prev, h_prev) # shape (batch_size*doc_len, extract_hidden_size)
            score_flat = self.score_linear_s( torch.tanh(self.score_linear_q(h_next) + self.score_linear_d(sent_enc_doc_flat)) ) #shape (batch_size*doc_len, )
            score_t = score_flat.contiguous().view(self.batch_size, self.doc_len) # (batch_size, doc_len)
            if sents_selected:
                mask = self.create_score_mask(sents_selected) # shape (batch_size, doc_len)
                # score_t.masked_fill_(mask, -float('inf')) # turn scores of selected sentences to -inf
                score_t.masked_fill_(mask, float(-1000000)) # turn scores of selected sentences to -1000000
            sent_selected_t = torch.argmax(score_t, dim=1) #shape (batch_size, ). Selected sentences at time step t
            sents_selected.append(sent_selected_t)
            sents_scores.append(score_t)

            # update hidden state and last selected sentence
            h_prev = h_next
            # s_prev_list = []
            # for num_doc, sent_idx in enumerate(sent_selected_t):
            #     print(sent_enc_doc[num_doc, sent_idx, :].size())
            #     print(sent_enc_doc[num_doc, sent_idx, :].repeat(self.doc_len, 1).size())
            #     s_prev_list.append( sent_enc_doc[num_doc, sent_idx, :].repeat(self.doc_len, 1) )
            #     print(s_prev_list)
            s_prev_list = [sent_enc_doc[num_doc, sent_idx, :].repeat(self.doc_len, 1) for num_doc, sent_idx in enumerate(sent_selected_t)]
            s_prev = torch.cat(s_prev_list, dim=0) #shape: (batch_size*doc_len, sent_hidden_size*2)

        sents_selected = torch.stack(sents_selected, dim=0).transpose(0,1) #shape (batch_size, T)
        sents_scores = torch.stack(sents_scores, dim=0).transpose(0,1) #shape (batch_size, T, doc_len)

        return sents_scores, sents_selected

    def forward(self, input):
        ''' sents scores and sents selected'''
        sent_enc_doc = self.encode(input)
        sents_scores, sents_selected = self.score_selection(sent_enc_doc)
        # print(sent_enc_doc.size()) #shape (batch_size, doc_len, sent_hidden_size*2)
        # print(sents_scores.size()) #shape (batch_size, T, doc_len)
        # print(sents_selected.size()) #shape (batch_size, T)
        # print(sent_enc_doc[1])
        # print(sents_scores[1])
        # print(sents_selected[1])
        return sents_scores, sents_selected
