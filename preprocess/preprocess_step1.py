import pickle
import re

# First step: clean up documents and break each document into sentences
### This is done here in this file

with open('newsroom_train_100000.p', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

summaries = data[0]
documents = data[1]

total = len(summaries)

def insert_end_sent_token(matchobj):
    if matchobj.group(0)[0] == '.':
        return matchobj.group(0).replace('.', '.<eos>')
    elif matchobj.group(0)[0] == '?':
        return matchobj.group(0).replace('?', '?<eos>')
    else:
        return matchobj.group(0).replace('!', '!<eos>')

def break_joint_word(matchobj):
    match_str = matchobj.group(0)
    return re.search('[A-Z]{2}', match_str).group(0) + ' ' + re.search('[a-z]+', match_str).group(0)

documents_sent_lvl = []
summaries_sent_lvl = []

for i in range(total):
    if i % int(total/10) == 0:
        print('Processing document', i)

    summary = summaries[i]
    document = documents[i]

    # There are summaries and/or documents that do not contain anything!
    if summary != None and len(summary) > 0:
        if document != None and len(document) > 0:

            # Replace certain substrings to make parsing easier
            document = document.replace('\n\n', '<eos>')
            document = document.replace(chr(8220), '"')
            document = document.replace(chr(8221), '"')
            document = document.replace(chr(8216), "'")
            document = document.replace(chr(8217), "'")
            document = document.replace("''", '"')
            document = document.replace('."', '".')
            document = document.replace('No.', 'No')
            document = document.replace('Mr.', 'Mr')
            document = document.replace('Mrs.', 'Mrs')

            # Turn abbreviations (that has periods) into <abbr> tokens
            document = re.sub('([A-Z][\.][\s]?)+', '<abbr> ', document)

            # Break uppercase-joint-with-lowercase words, e.g. LOHANmay and CARREYgot
            document = re.sub('[A-Z]{2}[a-z]+', break_joint_word, document)

            # Add a <eos> token at the end of each sentence
            document = re.sub('[\.\?!][\s]?[A-Z][^\.\?!]+', insert_end_sent_token, document)

            # Remove redundant spaces
            summary = summary.replace('  ', ' ')
            document = document.replace('  ', ' ')

            # Split an article into sentences
            document_split = document.split('<eos>')

            # Remove empty sentence and a possible space at the beginning of each non-empty sentence.
            document_split = list(filter(None, document_split))
            document_split = [sentence[1:] if sentence[0] == ' ' else sentence for sentence in document_split]

            documents_sent_lvl.append(document_split)
            summaries_sent_lvl.append(summary)

print('Total number of processed documents:', len(summaries_sent_lvl))

with open('newsroom_train_100000_sent_lvl.p', 'wb') as pickle_file:
    pickle.dump((summaries_sent_lvl, documents_sent_lvl), pickle_file)

# Second step: break each sentence into words
### This is done in preprocess_step2.py

# Third step: pad mini-batches of sentences and documents
### This is done in vocab.to_input_tensor()
