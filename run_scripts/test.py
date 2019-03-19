import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # Add parent folder to system path

import torch.optim as optim
import matplotlib.pyplot as plt
import random
import time
import numpy as np
from modules.objective import *
from modules.neusum_model import *
from modules.rouge_calc import *

def get_summary_tokens(sent_indices, doc_tokens, device):
    summary_tensor = torch.index_select(doc_tokens, 0, sent_indices)
    summary_tensor_flat = summary_tensor.view(-1)
    mask = torch.tensor([0 if x ==vocab['<pad>'] else 1 for x in summary_tensor_flat.tolist()], device=device)
    summary_tensor_flat = summary_tensor_flat * mask
    summary_tensor_flat = summary_tensor_flat[summary_tensor_flat.nonzero()].view(-1)
    summary_flat = summary_tensor_flat.tolist() # rouge_n can take in multiple references
    return summary_flat

def rouge_f1(hypothesis, reference, rouge_num=1):
    precision = rouge_n(hypothesis, reference, rouge_num, 0)
    recall = rouge_n(hypothesis, reference, rouge_num, 1)
    if precision > 0 and recall > 0:
        r = 2 / (1/precision + 1/recall)
    else:
        r = 0.0 # Do not just use 0, need a float type
    return r

## ============================== DEVICE SELECTION ============================== ##

device_name = 'cpu'
# device_name = 'cuda:0'
device = torch.device(device_name)

## ============================== LOAD PROCESSED DATA ============================== ##
## docs should have the form:
## (List[List[List[str]]])
## inner list = word
## outer list = sent
## outer outer = doc

with open('../dataset/TEST_ext_data_20_sents_doc_3_sents_summary_1000_samples.p', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

# orig_summaries = data[0]
# ext_summaries = data[2]
documents = data[1]
gold_sents_indices = torch.tensor(data[3], device=device)

vocab = Vocab.load('../dataset/newsroom_train_99844_vocab.json')

num_docs = len(documents)
print('Number of docs: %d' % num_docs)
print('Sentences per document:', max([len(documents[i]) for i in range(num_docs)]))
print('Gold summary sentence number:', len(data[3][0]))

## ============================== NET PARAMETER SPECIFICATIONS ============================== ##

word_embed_size = 50 # 50 (parameter from the paper)
hidden_size = 256 # 256
sent_hidden_size = 256 #256
extract_hidden_size = 256 #256

## ============================== TRAIN PARAMETER SPECIFICATIONS ============================== ##

batch_size = 5
rouge_num = 1
load_model = True
max_t_step = 3

## ============================== SOME SETTINGS BEFORE TRAINING ============================== ##

neusum = NeuSum(word_embed_size, hidden_size, sent_hidden_size, extract_hidden_size, vocab, device,
    sent_dropout_rate=0.0, doc_dropout_rate=0.0, max_t_step=max_t_step, attn_size=30)
neusum.eval()
neusum = neusum.to(device)

if load_model:
    load_from_epoch = 200
    train_name = 'ext_data_20_sents_doc_3_sents_summary_1000_samples'
    neusum.load_state_dict(torch.load('../saved_models/'+train_name+'/epoch_'+str(load_from_epoch)+'_neusum.p', map_location=device_name))
    # optimizer.load_state_dict(torch.load('../saved_models/'+train_name+'/epoch_'+str(load_from_epoch)+'_optimizer.optim'))

## ============================== START TRAINING LOOP ============================== ##

num_batches = num_docs // batch_size

rouge_num = 1
lead3_indices = torch.tensor([0,1,2], device=device)

sum_r = 0
sum_lead3_r = 0

for batch in range(num_batches):
    docs_batch = documents[batch*batch_size : (batch+1)*batch_size]
    gold_sent_indices_batch = gold_sents_indices[batch*batch_size:(batch+1)*batch_size, :]

    _, select_sent_indices_batch, doc_tokens_batch = neusum.forward(docs_batch)

    for i in range(batch_size):
        doc_tokens_i = doc_tokens_batch[i]
        gold_sent_indices_i = gold_sent_indices_batch[i]
        select_sent_indices_i = select_sent_indices_batch[i]

        reference = get_summary_tokens(gold_sent_indices_i, doc_tokens_i, device)
        hypothesis = get_summary_tokens(select_sent_indices_i, doc_tokens_i, device)
        lead3_hypothesis = get_summary_tokens(lead3_indices, doc_tokens_i, device)
        reference = [reference]

        r = rouge_f1(hypothesis, reference, rouge_num)
        lead3_r = rouge_f1(lead3_hypothesis, reference, rouge_num)

        sum_r += r
        sum_lead3_r += lead3_r

    print('Batch:', batch)
    print(sum_r/(batch+1)/batch_size)
    print(sum_lead3_r/(batch+1)/batch_size)

print(sum_r/num_docs)
print(sum_lead3_r/num_docs)
