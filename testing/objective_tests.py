import torch
import pickle
from models.vocab import Vocab
from models.rouge_calc import rouge_1, rouge_2, rouge_3, rouge_n
from models.objective import *
import numpy as np

# Test with toy dataset
with open('../toy_dataset/toy_dataset_word_lvl.p', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

summaries = data[0]
documents = data[1]

print(documents)

vocab = Vocab.load('../toy_dataset/toy_vocab.json')
device = torch.device('cpu')
doc_indices = vocab.to_input_tensor(documents, device)

print(doc_indices)

T = 3
batch_size, doc_len, _ = doc_indices.size()
print('batch_size: %d' % batch_size)
torch.manual_seed(0)
sents_scores = torch.rand(batch_size, T, doc_len, requires_grad=True) # has shape:(doc_len, batch_size, T)

pred_sents_indices = torch.tensor([[0,2,1], [2,1,0]], device=device) #shape (batch_size, T)
gold_sents_indices = torch.tensor([[0,2,1], [2,1,0]], device=device); # should also be: #shape (batch_size, T), right?

print(pred_sents_indices.shape), ## wrong shape?

P, logP = pred_dist(sents_scores); #we should figure out what this is
Q = label_dist(sents_scores, pred_sents_indices, gold_sents_indices, doc_indices, vocab, device)
print(P.shape, Q.shape)
print('P: ' + str(P))
print('Q:' + str(Q))
loss = KL(logP, Q)

print('P has shape:', P.size())
print('Q has shape:', Q.size())
print('KL loss is:', loss)
print(P)
print(Q)

## input to kldivloss is log probs and probs
kldiv = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean')

print(torch.rand((2,2)))
#log softmax appears like it will always yield a negative number?
n1 = torch.log_softmax(4*torch.rand((2,2)), dim = 1)
print(n1)
n2 = torch.randint(0, 1, (2,2)).float()
n2 = torch.from_numpy(np.array([[1,0],[0,1]])).float(); #representative of the fact that Q is mostly 0's or 1's
print(n2)
print(kldiv(torch.log(n1), n2))

## how can the KL divergence become infinite? IF Q has any zeros in it. or P has infinities in it?
Pi = torch.from_numpy(np.array([0.2, 0.1, 0.2, 0.4,])); #number's here have to be smaller than 1.
Qi = torch.from_numpy(np.array([0.01, 1, 0.1, 0.4])); # any zero's here sends the kldiv loss to zero.
print(torch.log(Qi))
print(kldiv(torch.log(Pi), Qi)) #is not infinity?
print(kldiv(torch.log(Qi), Pi))
#print(torch.log(torch.tensor([0]).float())); #any zeros cause an immediate infinity.

## extension uestion: how can we get a 0 log_softmax
print('investigation to zero in torch.log_softmax')
'''
this means that the input of log is 1, which is very hard to do with a softmax function...
'''
print(torch.log_softmax(torch.tensor([1-1e-12, 1e-12]).float(), dim = 0));

