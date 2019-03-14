import torch
from torch import nn;



"""
@param sent_enc_doc (Tensor): document-level encodings. Shape: (batch_size, doc_len, sent_hidden_size*2). Output from the encoder
@returns sent_scores (Tensor): sentence scores of every time step. Shape (batch_size, T, doc_len) where T=3 is the total number of time steps (max_t_step)
@returns sent_selected (Tensor): The indices of selected three sentences of each document. Shape (batch_size, 3)
"""

batch_size = 2;
doc_len = 2;
sent_hidden_size = 10;

sent_enc_doc = torch.randn(batch_size, doc_len, sent_hidden_size*2);


class sentenceScorer(torch.nn.Module):
    '''
    with just a sentence encoding, how does the sentence scorer know what is the "best"? backprop from the end FoM?
    '''
    def __init__(self):
        '''
        attn_size = attention?
        '''
        super(sentenceScorer, self).__init__()

        self.h0_linear = 0;
        self.ht_gru = 0
        self.score_linear_q = nn.Linear(extract_hidden_size, attn_size)
        self.score_linear_d = nn.Linear(sent_hidden_size*2, attn_size)
        self.score_linear_s = nn.Linear(attn_size, 1);
        #h_prev vs s_rev = S_{t-1} t=0 is the init and then we have t=1, t=2, t=3?


# sent_hidden_size = sent_enc_doc.view()[2] / 2
last_back_vec = sent_enc_doc[:, 0,sent_hidden_size:sent_hidden_size * 2]  # shape (batch_size, sent_hidden_size)
last_back_vec_flat = last_back_vec.repeat(1, doc_len).contiguous().view(-1, sent_hidden_size)  # shape (batch_size*doc_len, sent_hidden_size)

# initialization
h_prev = torch.tanh(self.h0_linear(last_back_vec_flat))  # shape (batch_size*doc_len, extract_hidden_size)
# h_prev is the last part of the MLP layer

sents_selected = []  # 'S0' in pdf
sents_scores = []
sent_selected_t = None
s_prev = torch.zeros(self.batch_size * self.doc_len, self.sent_hidden_size * 2)  # 's0' in pdf
sent_enc_doc_flat = sent_enc_doc.contiguous().view(-1,
                                                   self.sent_hidden_size * 2)  # shape (batch_size*doc_len, extract_hidden_size)

for t in range(self.max_t_step):  # max_t_step will typically be three. (for a 3 sentence summary)
    h_next = self.ht_gru(s_prev, h_prev)  # shape (batch_size*doc_len, extract_hidden_size)
    score_flat = self.score_linear_s(torch.tanh(
        self.score_linear_q(h_next) + self.score_linear_d(sent_enc_doc_flat)))  # shape (batch_size*doc_len, )
    # beyond the previous line, we now have the sentence score.
    score_t = score_flat.contiguous().view(self.batch_size, self.doc_len)  # (batch_size, doc_len)
    if sents_selected:
        mask = self.create_score_mask(sents_selected)  # shape (batch_size, doc_len)
        score_t.masked_fill_(mask, -float(
            'inf'))  # turn scores of selected sentences to -inf, so they can NEVER be selected in next step
    sent_selected_t = torch.argmax(score_t, dim=1)  # shape (batch_size, ). Selected sentences at time step t
    sents_selected.append(sent_selected_t)

    # it appears that the sent scores are the output of a linear layer so the scores could be neg or pos real values
    sents_scores.append(score_t)

    # update hidden state and last selected sentence
    h_prev = h_next
    # s_prev_list = []
    # for num_doc, sent_idx in enumerate(sent_selected_t):
    #     print(sent_enc_doc[num_doc, sent_idx, :].size())
    #     print(sent_enc_doc[num_doc, sent_idx, :].repeat(self.doc_len, 1).size())
    #     s_prev_list.append( sent_enc_doc[num_doc, sent_idx, :].repeat(self.doc_len, 1) )
    #     print(s_prev_list)
    s_prev_list = [sent_enc_doc[num_doc, sent_idx, :].repeat(self.doc_len, 1) for num_doc, sent_idx in
                   enumerate(sent_selected_t)]
    s_prev = torch.cat(s_prev_list, dim=0)  # shape: (batch_size*doc_len, sent_hidden_size*2)

sents_selected = torch.stack(sents_selected, dim=0).transpose(0, 1)  # shape (batch_size, T)
sents_scores = torch.stack(sents_scores, dim=0).transpose(0, 1)  # shape (batch_size, T, doc_len)
