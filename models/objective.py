import torch
import pickle
from vocab import Vocab
from rouge_calc import rouge_1, rouge_2, rouge_3, rouge_n

# Objective function
# TODO:
# 1. Construct model prediction distribution P
# 2. Construct label distribution Q
# 3. Compute KL divergence loss KL(P||Q)

def pred_dist(sents_scores):
    """
    @param sents_scores (Tensor): sentence scores of every time step
        Shape (batch_size, T, doc_len) where T=3 is the total number of time steps
    """
    P = torch.softmax(sents_scores, dim=2) # Shape (batch_size, T, doc_len)
    P = P.permute(2,0,1) # Shape (doc_len, batch_size, T)
    return P

def label_dist(sents_scores, pred_sents_indices, gold_sents_indices, docs, vocab, device, temperature=20):
    """
    @param sents_scores (Tensor): sentence scores of every time step
        Shape (batch_size, T, doc_len) where T=3 is the total number of time steps
    @param pred_sents_indices (Tensor): The indices of selected T sentences of each document
        Shape (batch_size, T)
    @param docs (Tensor): batch of documents as a tensor of tokens (int)
    @param vocab (Vocab): vocabulary object
    @param device (torch.device): torch.device('cpu') or torch.device('CUDA:0')
    @param temperature (float): smooth coefficient in calculating Q, default = 20
    """
    batch_size, T, doc_len = sents_scores.size()
    Q_list = []
    for t in range(T):
        tminus1_pred_sents_indices = pred_sents_indices[:,:t] # Shape (batch_size, t)

        t_pred_sents_indices = [[tminus1_pred_sents_indices.tolist()[b] + [i] for i in range(doc_len)] for b in range(batch_size)]
        t_pred_sents_indices = torch.tensor(t_pred_sents_indices, device=device) # Shape (batch_size, doc_len, t+1)

        rouge_t = rouge(t_pred_sents_indices, gold_sents_indices, docs, vocab, device) # Rouge scores of shape (batch_size, doc_len)

        if t > 0:
            rouge_tminus1 = rouge(tminus1_pred_sents_indices.view(batch_size, 1, t), gold_sents_indices, docs, vocab, device) # Rouge scores of shape (batch_size, 1)
            g = rouge_t - rouge_tminus1 # Rouge score differences of shape (batch_size, doc_len)
        else:
            g = rouge_t
        g_min = g.min(dim=1, keepdim=True)[0]
        g_max = g.max(dim=1, keepdim=True)[0]
        g_t = (g - g_min) / (g_max - g_min)

        Q_t = torch.softmax(g_t * temperature, dim=1) # Shape (batch_size, doc_len)
        Q_list.append(Q_t.unsqueeze(dim=0))

    Q = torch.cat(Q_list) # Shape (T, batch_size, doc_len)
    Q = Q.permute(2,1,0) # Shape (doc_len, batch_size, T)
    return Q

def rouge(pred_sents_indices, gold_sents_indices, docs, vocab, device):
    """
    @param pred_sents_indices (Tensor): predicted sentence indices
        Shape (batch_size, num_choices_of_summary, pred_summary_len)
    @param gold_sents_indices (Tensor): golden sentence indices
        Shape (batch_size, gold_summary_len)
    @param docs (Tensor): batch of documents as a tensor of tokens (int)
    @param vocab (Vocab): vocabulary object
    @param device (torch.device): torch.device('cpu') or torch.device('CUDA:0')
    """
    batch_size, num_choices_of_summary, _ = pred_sents_indices.size()
    r_list_list = []

    for batch in range(batch_size):
        doc = docs[batch]
        reference = torch.index_select(doc, 0, gold_sents_indices[batch])
        reference = reference.view(-1).tolist() # Flatten out sentences into one list
        reference = list(filter(lambda x: x!=vocab['<pad>'], reference)) # Remove pad tokens
        reference = [reference] # rouge_n can take in multiple references

        r_list = []
        for i in range(num_choices_of_summary):
            hypothesis = torch.index_select(doc, 0, pred_sents_indices[batch,i])
            hypothesis = hypothesis.view(-1).tolist() # Flatten out sentences into one list
            hypothesis = list(filter(lambda x: x!=vocab['<pad>'], hypothesis)) # Remove pad tokens

            # Use reference and hypothesis to calculate rouge score
            r = rouge_1(hypothesis, reference, 0)
            r_list.append(r)

            '''
            print('Reference is:', reference)
            print('Hypothesis is:', hypothesis)
            print('Rouge score is:', r)
            print('')
            '''
        r_list_list.append(r_list)

    return torch.tensor(r_list_list, device=device)

def KL(P, Q):
    """
    Calculate KL divergence loss between predicted distribution P and labeled distribution Q
    """
    return torch.nn.KLDivLoss(reduction='sum')(P, Q)

# Test with toy dataset
with open('toy_dataset_word_lvl.p', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

summaries = data[0]
documents = data[1]

vocab = Vocab.load('toy_vocab.json')
device = torch.device('cpu')
doc_indices = vocab.to_input_tensor(documents, device)

T = 3
batch_size, doc_len, _ = doc_indices.size()
torch.manual_seed(0)
sents_scores = torch.rand(batch_size, T, doc_len)

pred_sents_indices = torch.tensor([[0,2,1], [2,1,0]], device=device)
gold_sents_indices = torch.tensor([[1,2], [0,1]], device=device)

P = pred_dist(sents_scores)
Q = label_dist(sents_scores, pred_sents_indices, gold_sents_indices, doc_indices, vocab, device)
loss = KL(P, Q)

print('P has shape:', P.size())
print('Q has shape:', Q.size())
print('KL loss is:', loss)
