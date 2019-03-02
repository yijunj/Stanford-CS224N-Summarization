import gensim
import time
# import warnings
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# Load Google's pre-trained Word2Vec model.
tic = time.time()
model = gensim.models.KeyedVectors.load_word2vec_format('./Embedding/GoogleNews-vectors-negative300.bin', binary=True)
toc = time.time() # Takes ~1min on my laptop

print('Word2vec model loaded. Time spent in seconds:', toc - tic)

vocab = model.wv.vocab # The vocabulary (as a python dictionary)

print(model.wv['do']) # Look up embeddings like this
