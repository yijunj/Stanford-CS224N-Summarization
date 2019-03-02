import pickle
import re

with open('newsroom_train_100000.p', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

summaries = data[0]
documents = data[1]

total = len(summaries)

def insert_end_sent_space(matchobj):
    if matchobj.group(0)[0] == '.':
        return matchobj.group(0).replace('.', '.<EOS>')
    elif matchobj.group(0)[0] == '?':
        return matchobj.group(0).replace('?', '?<EOS>')
    else:
        return matchobj.group(0).replace('!', '!<EOS>')

def break_joint_word(matchobj):
    match_str = matchobj.group(0)
    return re.search('[A-Z]{2}', match_str).group(0) + ' ' + re.search('[a-z]+', match_str).group(0)

# First step: clean up documents and break each document into sentences
clean_documents = []
clean_summaries = []

for i in range(total):
    if i % int(total/10) == 0:
        print('Processing document', i)

    summary = summaries[i]
    document = documents[i]

    # There are summaries and/or documents that do not contain anything!
    if summary != None and len(summary)>0:
        if document != None and len(document)>0:

            # Replace certain substrings to make parsing easier
            document = document.replace('\n\n', '<EOS>')
            document = document.replace(chr(8220), '"')
            document = document.replace(chr(8221), '"')
            document = document.replace(chr(8216), "'")
            document = document.replace(chr(8217), "'")
            document = document.replace("''", '"')
            document = document.replace('."', '".')
            document = document.replace('No.', 'No')
            document = document.replace('Mr.', 'Mr')
            document = document.replace('Mrs.', 'Mrs')

            # Turn abbreviations (that has periods) into <ABBR> tokens
            document = re.sub('([A-Z][\.][\s]?)+', '<ABBR> ', document)

            # Break uppercase-joint-with-lowercase words, e.g. LOHANmay and CARREYgot
            document = re.sub('[A-Z]{2}[a-z]+', break_joint_word, document)

            # Add a <EOS> token at the end of each sentence
            document = re.sub('[\.\?!][\s]?[A-Z][^\.\?!]+', insert_end_sent_space, document)

            # Remove redundant spaces
            summary = summary.replace('  ', ' ')
            document = document.replace('  ', ' ')

            # Split an article into sentences
            document_split = document.split('<EOS>')

            # Remove empty sentence and a possible space at the beginning of each non-empty sentence.
            document_split = list(filter(None, document_split))
            document_split = [sentence[1:] if sentence[0] == ' ' else sentence for sentence in document_split]

            clean_documents.append(document_split)
            clean_summaries.append(summary)

print('Total number of processed documents:', len(clean_summaries))

with open('newsroom_train_100000_clean.p', 'wb') as pickle_file:
    pickle.dump((clean_summaries, clean_documents), pickle_file)

# Second step: break each sentence into words

# Third step: for each sentence, get a list of word embeddings

# Fourth step: feed these word embeddings into a sentence encoder to get a sentence encoding
