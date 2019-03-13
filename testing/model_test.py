
import sys
sys.path.append('D:\\Documents\\Classes\\CS224n\\project')
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np
import random
import math
import os
import time
import pickle
from models.vocab import *

from models.sent_encoder import *
from models.doc_encoder import *

# Test with toy dataset
with open('../toy_dataset/toy_dataset_word_lvl.p', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

summaries = data[0]
documents = data[1]
print(summaries);
print(documents)

vocab = Vocab.load('../toy_dataset/toy_vocab.json')
device = torch.device('cpu')

word_embed_size = 300;
hidden_size = 20;
sent_hidden_size = 2*hidden_size; #should be fixed by the sentence encoder right, so it really shouldn't be a free parameter

#vocab = None; #vocab object
num_words = 1000;

docLength = 10;
b = 20; #total sentences over each docuemnt.

## randBatch is a batch of sentences from different documents... each batch has sentences from different docs..

randBatch = torch.randint(0,len(vocab),(docLength,b))
print(randBatch)
source_lengths = [10 for i in range(b)]
#input to se is only a word vector... but we feed in a sequence

se = SentenceEncoder(word_embed_size, hidden_size, vocab)
de = DocumentEncoder(hidden_size, sent_hidden_size)

#padding because each batch consists of a group of sentences, padded out

#source_padded: (src_len, b, word_embed_size)
#source lengths: List of actual lengths for each of the source sentences in the batch

doc_batch_size = 2;
sent_enc = se.forward(randBatch, source_lengths);
print(sent_enc.shape)
print(sent_enc)
#shape (batch size, 2*hidden size)
sent_enc_decoded = sent_enc.view( doc_batch_size, docLength, sent_hidden_size);

#input shape for the doc encoder.
#(doc_len, batch_size, sent_embed_size)
#doc_lengths = num_sentences, so it is just source lengths?
#de input: (doc_len, batch_size, sent_embed_size)

#every sentence encoding is what

doc_lengths = [10,10]
doc_enc = de.forward(sent_enc_decoded,  doc_lengths)
print(doc_enc.shape)

print(doc_enc)