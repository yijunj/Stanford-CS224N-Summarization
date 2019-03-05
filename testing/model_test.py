
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

from models.sent_encoder import *
from models.doc_encoder import *


word_embed_size = 300;
hidden_size = 20;
sent_hidden_size = 2*hidden_size; #should be fixed by the sentence encoder right, so it really shouldn't be a free parameter

vocab = None; #vocab object
num_words = 1000;

np.random.seed(42)
fake_embedding = np.random.randn(num_words, word_embed_size)
# def fake_corpus(num_docs, max_doc_length = 10):
#     for i in range(num_docs):
#         2;
#     return;

docLength = 10;
b = 2
randBatch = torch.randn(10, b, word_embed_size)
source_lengths = [10 for i in range(b)]
#input to se is only a word vector... but we feed in a sequence

se = SentenceEncoder(word_embed_size, hidden_size, vocab)

de = DocumentEncoder(hidden_size, sent_hidden_size)

#padding because each batch consists of a group of sentences, padded out

#source_padded: (src_len, b, word_embed_size)
#source lengths: List of actual lengths for each of the source sentences in the batch

sent_enc = se.forward(randBatch, source_lengths);
print(sent_enc.shape)

#doc_lengths = num_sentences, so it is just source lengths?
#de input: (doc_len, batch_size, sent_embed_size)

#every sentence encoding is what

doc_enc = de.forward(sent_enc, source_lengths)
print(doc_enc.shape)
