import torch.optim as optim
import matplotlib.pyplot as plt
import random
import time
import numpy as np
from modules.objective import *
from modules.neusum_model import *

## ============================== LOAD PROCESSED DATA ============================== ##
## docs should have the form:
## (List[List[List[str]]])
## inner list = word
## outer list = sent
## outer outer = doc

with open('./dataset/extraction_summaries_2_tiny.p', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

# orig_summaries = data[0]
# ext_summaries = data[2]
documents = data[1]
gold_sents_indices = torch.from_numpy(np.array(data[3])).long()

vocab = Vocab.load('./dataset/newsroom_train_99844_vocab.json')

num_docs = len(documents)
print('Num of docs: %d' % num_docs)

## ============================== NET PARAMETER SPECIFICATIONS ============================== ##

word_embed_size = 30 # 50 (parameter from the paper)
hidden_size = 5 # 256
sent_hidden_size = 5 #256
extract_hidden_size = 5 #256

## ============================== TRAIN PARAMETER SPECIFICATIONS ============================== ##

device = torch.device('cpu')
# device = torch.device('cuda:0')
batch_size = 5
num_epochs = 10000
print_every_epoch = 10
save_every_epoch = 100
load_model = False

## ============================== SOME SETTINGS BEFORE TRAINING ============================== ##

neusum = NeuSum(word_embed_size, hidden_size, sent_hidden_size, extract_hidden_size, vocab, device,
    sent_dropout_rate=0.0, doc_dropout_rate=0.0, max_t_step=3, attn_size=30)
neusum.train()
neusum = neusum.to(device)

optimizer = optim.Adam(neusum.parameters(), lr = 0.0005)

if load_model:
    load_from_epoch = 1
    neusum.load_state_dict(torch.load('./saved_models/epoch_' + str(load_from_epoch) + 'neusum.p'))
    optimizer.load_state_dict(torch.load('./saved_models/epoch_' + str(load_from_epoch) + '_optimizer.optim'))

# neusum.zero_grad()

## ============================== START TRAINING LOOP ============================== ##

loss_history = []
num_batches = num_docs // batch_size

data_indices_list = np.arange(num_docs)

tic = time.time()
for epoch in range(num_epochs):
    epoch_loss = 0
    np.random.shuffle(data_indices_list) # Randomly shuffle data

    for batch in range(num_batches):
        data_indices_batch = data_indices_list[batch*batch_size : (batch+1)*batch_size]
        docs_batch = [documents[i] for i in data_indices_batch]
        gold_sent_indices_batch = torch.index_select(gold_sents_indices, 0, torch.from_numpy(data_indices_batch).long())

        # docs_batch = documents[batch*batch_size : (batch+1)*batch_size]
        # gold_sent_indices_batch = gold_sents_indices[batch*batch_size:(batch+1)*batch_size, :]

        optimizer.zero_grad()
        select_sent_scores_batch, select_sent_indices_batch, doc_tokens_batch = neusum.forward(docs_batch)

        loss, _ = neusum_loss(select_sent_scores_batch, select_sent_indices_batch, gold_sent_indices_batch, doc_tokens_batch, vocab, device)
        epoch_loss += loss

        loss.backward()
        optimizer.step()

    epoch_loss_per_doc = epoch_loss / batch_size / num_batches
    loss_history.append(epoch_loss_per_doc)

    if epoch % print_every_epoch == 0 or epoch == num_epochs - 1:
        toc = time.time()
        print('')
        print('============ Epoch %d ============' % epoch)
        print('Time: %f s' % (toc - tic))
        print('Loss: %f' % epoch_loss_per_doc)

    if epoch % save_every_epoch == 0 or epoch == num_epochs - 1:
        torch.save(neusum.state_dict(), './saved_models/epoch_' + str(epoch) + '_neusum.p' )
        # torch.save(optimizer.state_dict(), '../saved_models/epoch_' + str(epoch) + '_optimizer.optim')
        print('------------')
        print('Model and optimizer saved')

        fig = plt.figure()
        plt.plot(loss_history)
        fig.savefig('./saved_figures/loss_til_epoch_' + str(epoch) + '.png', dpi=fig.dpi)
        plt.close(fig)
        print('Loss figure saved')

        with open('./loss.p', 'wb') as pickle_file:
            pickle.dump(loss_history, pickle_file)
