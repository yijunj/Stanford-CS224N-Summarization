import sys
sys.path.append('D:\\Documents\\Classes\\CS224n\\project')
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np
import pickle
from models.vocab import *
from models.objective import *
from models.sent_encoder import *
from models.doc_encoder import *
from models.NeuSum_model import *
# Test with toy dataset
with open('../toy_dataset/toy_dataset_word_lvl.p', 'rb') as pickle_file:
    data = pickle.load(pickle_file)


summaries = data[0]
documents = data[1]
print(summaries);
print(documents)

vocab = Vocab.load('../toy_dataset/toy_vocab.json')
device = torch.device('cpu')

doc_indices = vocab.to_input_tensor(documents, device)

print(doc_indices)

T = 3
batch_size, doc_len, _ = doc_indices.size()
gold_sents_indices = torch.tensor([[1,2], [0,1]], device=device)



## =============================== PARAMETER SPECIFICATIONS ==============================================##
word_embed_size = 300;
hidden_size = 20;
sent_hidden_size = 2*hidden_size; #should be fixed by the sentence encoder right, so it really shouldn't be a free parameter

num_docs = 2;
docLength = 10;
b = 20; #total sentences over each docuemnt.
extract_hidden_size = 10;

Nsum=NeuSum(word_embed_size, hidden_size, sent_hidden_size, extract_hidden_size, vocab, device, \
            sent_dropout_rate=0.3, doc_dropout_rate=0.2, max_t_step = 3, attn_size = 512);

## random data;
randBatch = torch.randint(0,len(vocab),(b, docLength))
#print(randBatch)
gold_sents_indices = torch.randint(0, 10, (num_docs, ))

#to_input_tensor(self, docs: List[List[List[str]]], device: torch.device) -> torch.Tensor:
        # """ Convert list of sentences (words) into tensor with necessary padding for
        # shorter sentences.


doc_indices = randBatch;

source_lengths = [10 for i in range(b)]

#(List[List[List[str]]])
#inner list = word
#outer list = sent
#outer outer = doc
num_docs = 2;
num_sentences = 2; #or doc_len
doc_len = num_sentences;
padded_sent_len = 4;
rand_input = [[['ab' for i in range(padded_sent_len)] for j in range(doc_len)] for k in range(num_docs)];
print(rand_input)

#
# sent_enc_doc = Nsum.encode(rand_input);
# # encoded sentences on doc-level, shape (batch_size, doc_len, sent_hidden_size*2)
# print(sent_enc_doc.shape)
#
# #score selection: The indices of selected three sentences of each document. Shape (batch_size, 3)
# sent_scores, score_select_indices = Nsum.score_selection(sent_enc_doc);
# print(sent_scores.shape)
# print(score_select_indices.shape)

sent_scores, score_select_indices = Nsum.forward(rand_input);
print('sent scores: '+str(sent_scores))          # has gradient
print(score_select_indices) #has the gradient

## evaluate objective # needs gradient
loss = neusum_loss(sent_scores, score_select_indices, gold_sents_indices, doc_indices, vocab, device);
print(loss);

optimizer = optim.Adam(Nsum.parameters(), lr = 0.0001)
loss.backward();
optimizer.step();



