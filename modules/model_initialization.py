import os
import sys

sys.path.append("D:\\Documents\\Classes\\CS224n\\project")

from models.vocab import *;
from models.NeuSum_model import *

import spacy
import pickle

word_embed_size, 
hidden_size, 
sent_hidden_size, 
extract_hidden_size


a =  NeuSum(word_embed_size, hidden_size, sent_hidden_size, extract_hidden_size, vocab, device,\
 		sent_dropout_rate=0.3, doc_dropout_rate=0.2, max_t_step = 3, attn_size = 512):
        """
