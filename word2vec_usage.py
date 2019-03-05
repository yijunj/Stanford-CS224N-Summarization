import gensim
import time
# import warnings
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# Load Google's pre-trained Word2Vec model.
tic = time.time()
pretrained_embeddings_path = "./Embedding/GoogleNews-vectors-negative300.bin.gz"
model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=True)
toc = time.time() # Takes ~1min on my laptop, took 77 seconds for me.
print('Word2vec model loaded. Time spent in seconds:', toc - tic)

vocab = model.wv.vocab # The vocabulary (as a python dictionary)

print(model.wv['do']) # Look up embeddings like this
