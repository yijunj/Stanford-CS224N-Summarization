import sys
sys.path.append('/Users/yiliu/Desktop/CS224N/final_project/scripts')

import torch.optim as optim
import matplotlib.pyplot as plt
#from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import random
import time
import numpy as np
from models.objective import *
from models.NeuSum_model import *

## ================================= LOAD PROCESSED DATA ====================================== ##
## docs should have the form:
# #(List[List[List[str]]])
#inner list = word
#outer list = sent
#outer outer = doc


# # Test with toy dataset
# with open('../dataset/newsroom_train_10_word_lvl.p', 'rb') as pickle_file:
#     data = pickle.load(pickle_file)
# summaries = data[0]
# documents = data[1]
# print(summaries)
# print(documents)

with open('../dataset/extraction_dataset_491.p', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

orig_summaries = data[0]
ext_summaries = data[2]
documents = data[1]
print(np.array(data[-1]))
gold_sents_indices = torch.from_numpy(np.array(data[-1])).long();
print(gold_sents_indices.shape)
# print(summaries)
# print(documents)
#print(documents[0])

vocab = Vocab.load('../dataset/newsroom_train_99844_vocab.json')

device = torch.device('cpu')
#device = torch.device('cuda:0')

# print('use device: %s' % device, file=sys.stderr)

## doc indices...for each document, put out the indices for word2vec of every sentence.
doc_indices = vocab.to_input_tensor(documents, device)
# print(doc_indices)
print('number of docs: %d' % len(documents))

batch_size, doc_len, _ = doc_indices.size()
print('batch_size: %d, doc_len: %d' %(batch_size, doc_len))
# gold_sents_indices = torch.tensor([[0,1], [0,1]], device=device)
# # gold_sents_indices = torch.tensor([[0,1,2], [2,0,1]], device=device)
# gold_ind = [random.sample(range(5), 3) for i in range(10)]
# gold_sents_indices = torch.tensor(gold_ind, device = device)

num_docs = len(documents)

## =============================== NET PARAMETER SPECIFICATIONS ==============================================##
word_embed_size = 30 # 50 (parameter from the paper)
hidden_size = 20 # 256
sent_hidden_size = 20 #256
extract_hidden_size = 10 #256

n_epochs = 500

Nsum=NeuSum(word_embed_size, hidden_size, sent_hidden_size, extract_hidden_size, vocab, device, \
            sent_dropout_rate=0.0, doc_dropout_rate=0.0, max_t_step = 3, attn_size = 30)
Nsum.train()
Nsum = Nsum.to(device)

## ===========================BENCHMARK: LOSS OF GOLDEN SUMMARY================================================
# sent_scores, score_select_indices = Nsum.forward(summaries);
# loss, (logP, P, Q) = neusum_loss(sent_scores, score_select_indices, gold_sents_indices, doc_indices, vocab, device);
# print('gold loss: '+str(loss))
## ============================TRAINING PARAMETERS =====================================
optimizer = optim.Adam(Nsum.parameters(), lr = 0.0005)
# Nsum.zero_grad()
## ====================== ACTUAL TRAINING LOOP ====================================

loss_history = []

tic = time.time()
batch_size = 20;
print(len(documents))
num_batches = len(documents)//batch_size;
print(num_batches)
for iter in range(n_epochs):
    epoch_loss = 0;
    for n in range(num_batches):

        doc_batch = documents[n*(batch_size):(n+1)*batch_size];
        gold_inds_batch = gold_sents_indices[n*(batch_size):(n+1)*batch_size,:]
        optimizer.zero_grad()
        sent_scores, sent_select_indices = Nsum.forward(doc_batch)
        # score_select_indices = torch.tensor([[0,0,0], [0,0,0]], device=device)
        # print(score_select_indices);
        # print(gold_sents_indices)

        #print(sent_scores.shape, score_select_indices.shape)
        #print('sent scores: '+str(sent_scores))          # has gradient

        ## ================= test sent scores ===================##
        # print('output sent scores: '+str(sent_scores)+' shape: '+str(sent_scores.shape)) ## should sent scores every be negative?
        ## =====================================================

        # print(score_select_indices) #has the gradien
        ## evaluate objective # needs gradient
        # torch.manual_seed(5)
        # sent_scores = torch.rand(2,3,3)
        # print(sent_scores)
        # score_select_indices = torch.tensor([[0,1,2], [2,1,0]], device = device)
        # gold_sents_indices = torch.tensor([[1,2,0], [0,2,1]], device = device)

        # print(sent_scores)
        # print(score_select_indices)
        # print(gold_sents_indices)
        # sent_select_indices = gold_sents_indices
        loss, (logP,P,Q) = neusum_loss(sent_scores, sent_select_indices, gold_inds_batch, doc_indices, vocab, device)
        epoch_loss+=loss;
        # print('P: '+str(P))
        # print('logP: '+str(logP))
        # print('Q:' +str(Q))
        # print(loss)
        # print('----------')
        loss_history.append(loss)

        loss.backward();
        optimizer.step();
    if iter % 1 == 0 :
        toc = time.time()
        print('============Epoch ', iter, '=============')
        print('Time: ', toc - tic)
        print('epoch_loss: %d'% epoch_loss)

for i in range(5):
    print('=============Document ', i, '===============')
    print('selected indices is', sent_select_indices[i].tolist())
    print('gold indices is', gold_sents_indices[i].tolist())
    print('selected scores is', sent_scores[i].tolist())

## ============================== removable tests for KL div ===================================##
kldiv = torch.nn.KLDivLoss(reduction = 'sum')
#print(kldiv(P,Q))


## ============================ POST TRAINING ==============================================##

torch.save(Nsum.state_dict(), 'neusum.pth')

plt.plot(loss_history)
plt.show()
