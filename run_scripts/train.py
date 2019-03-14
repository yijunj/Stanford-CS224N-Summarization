import sys
sys.path.append('D:\\Documents\\Classes\\CS224n\\project')

import torch.optim as optim
import matplotlib.pyplot as plt
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

from models.objective import *
from models.NeuSum_model import *

## ================================= LOAD PROCESSED DATA ====================================== ##
## docs should have the form:
# #(List[List[List[str]]])
#inner list = word
#outer list = sent
#outer outer = doc


# Test with toy dataset
with open('../toy_dataset/toy_dataset_word_lvl.p', 'rb') as pickle_file:
    data = pickle.load(pickle_file)


summaries = data[0]
documents = data[1]
print(summaries);
print(documents)

vocab = Vocab.load('../toy_dataset/toy_vocab.json')
device = torch.device('cpu')

## doc indices...for each document, put out the indices for word2vec of every sentence.
doc_indices = vocab.to_input_tensor(documents, device)
print(doc_indices)


batch_size, doc_len, _ = doc_indices.size()
print('batch_size: %d, doc_len: %d' %(batch_size, doc_len))
gold_sents_indices = torch.tensor([[1,2], [0,1]], device=device)
gold_sents_indices = torch.tensor([[0,1,2], [2,0,1]], device=device)

num_docs = len(documents);

## =============================== NET PARAMETER SPECIFICATIONS ==============================================##
word_embed_size = 30;
hidden_size = 20;
sent_hidden_size = 2*hidden_size; #should be fixed by the sentence encoder right, so it really shouldn't be a free parameter
extract_hidden_size = 10;

Nsum=NeuSum(word_embed_size, hidden_size, sent_hidden_size, extract_hidden_size, vocab, device, \
            sent_dropout_rate=0.0, doc_dropout_rate=0.0, max_t_step = 3, attn_size = 512);

## ===========================BENCHMARK: LOSS OF GOLDEN SUMMARY================================================
# sent_scores, score_select_indices = Nsum.forward(summaries);
# loss, (logP, P, Q) = neusum_loss(sent_scores, score_select_indices, gold_sents_indices, doc_indices, vocab, device);
# print('gold loss: '+str(loss))
## ============================TRAINING PARAMETERS =====================================
n_epochs = 100;
optimizer = optim.Adam(Nsum.parameters(), lr = 1e-10)
Nsum.zero_grad();
## ====================== ACTUAL TRAINING LOOP ====================================

loss_history = [];
for iter in range(n_epochs):
    optimizer.zero_grad();
    sent_scores, score_select_indices = Nsum.forward(documents);
    # print(score_select_indices);
    # print(gold_sents_indices)

    #print(sent_scores.shape, score_select_indices.shape)
    #print('sent scores: '+str(sent_scores))          # has gradient

    ## ================= test sent scores ===================##
    # print('output sent scores: '+str(sent_scores)+' shape: '+str(sent_scores.shape)) ## should sent scores every be negative?
    ## =====================================================

    # print(score_select_indices) #has the gradien
    ## evaluate objective # needs gradient

    loss, (logP,P,Q) = neusum_loss(sent_scores, score_select_indices, gold_sents_indices, doc_indices, vocab, device);
    # print('P: '+str(P))
    # print('logP: '+str(logP))
    # print('Q:' +str(Q))
    print(loss);
    loss_history.append(loss);

    loss.backward();
    optimizer.step();


## ============================== removable tests for KL div ===================================##
kldiv = torch.nn.KLDivLoss(reduction = 'sum');
#print(kldiv(P,Q))


## ============================ POST TRAINING ==============================================##

torch.save(Nsum.state_dict(), 'neusum.pth')

plt.plot(loss_history)
plt.show();