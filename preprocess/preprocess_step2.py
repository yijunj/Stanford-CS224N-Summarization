import pickle
import re

# First step: clean up documents and break each document into sentences

# Second step: break each sentence into words

with open('newsroom_train_99844_sent_lvl.p', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

summaries_sent_lvl = data[0]
documents_sent_lvl = data[1]

total = len(summaries_sent_lvl)

summaries_word_lvl = []
documents_word_lvl = []

for i in range(total):
    if i % int(total/10) == 0:
        print('Processing document', i)

    summary = summaries_sent_lvl[i]
    document = documents_sent_lvl[i]

    summary_split = summary.split(' ')
    summary_split = [re.sub("[^a-z0-9'<>]", '', word.lower()) for word in summary_split]
    summary_split = list(filter(None, summary_split))
    summaries_word_lvl.append(summary_split)

    document_word_lvl = []
    for sentence in document:
        sentence_split = sentence.split(' ')
        sentence_split = [re.sub("[^a-z0-9'<>]", '', word.lower()) for word in sentence_split]
        sentence_split = list(filter(None, sentence_split))
        document_word_lvl.append(sentence_split)
    documents_word_lvl.append(document_word_lvl)

print('Total number of processed documents:', len(summaries_word_lvl))

with open('newsroom_train_100000_word_lvl.p', 'wb') as pickle_file:
    pickle.dump((summaries_word_lvl, documents_word_lvl), pickle_file)

# Third step: pad mini-batches of sentences and documents

# Fourth step: for each sentence, get a list of word embeddings
