
## test the tokenization of different texts
# once we tokenize, we should be able to insert into the vocab object.

import os
import sys

sys.path.append("D:\\Documents\\Classes\\CS224n\\project")

from models.vocab import *;
import spacy
import pickle

spacy_en = spacy.load('en')


def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


## import sample dataset
filename = os.path.join("D:\\Documents\\Classes\\CS224n\\project", \
	'processed_train\\newsroom_train_100000.p');

(text, summaries) = pickle.load(open(filename, 'rb'));
print(len(text))

sample_text = text[0];
sample_tokens = tokenizer(sample_text);

print(sample_tokens)

## process num_samples

num_samples = 10000;
vocab = set();
for i in range(num_samples):
	if(not text[i]):
		continue;
	words = tokenizer(text[i]);
	vocab.update(words);
print(len(vocab))

word2idx = dict(zip(vocab, range(0, len(vocab))))
word2idx['<unk>'] = num_samples+1;
print(word2idx)

## we need to include a unk index ...

vocabObj = Vocab(word2idx);


## beyond this, what do we do, try to do a simple forward pass?