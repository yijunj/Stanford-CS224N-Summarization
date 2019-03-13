import torch
import pickle
from models.vocab import Vocab
from models.rouge_calc import rouge_1, rouge_2, rouge_3, rouge_n
from models.objective import *

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
torch.manual_seed(0)
sents_scores = torch.rand(batch_size, T, doc_len, requires_grad=True)

pred_sents_indices = torch.tensor([[0,2,1], [2,1,0]], device=device)
gold_sents_indices = torch.tensor([[1,2], [0,1]], device=device)

P = pred_dist(sents_scores)
Q = label_dist(sents_scores, pred_sents_indices, gold_sents_indices, doc_indices, vocab, device)

loss = KL(P, Q)

print('P has shape:', P.size())
print('Q has shape:', Q.size())
print('KL loss is:', loss)
print(P)
print(Q)