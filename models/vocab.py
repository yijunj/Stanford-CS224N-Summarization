#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODIFIED from

CS224N 2018-19: Homework 4
vocab.py: Vocabulary Generation
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

from collections import Counter
from docopt import docopt
from itertools import chain
import json
import torch
from typing import List
from utils import read_corpus, pad_sents


class Vocab(object):
    """ Vocabulary, i.e. structure containing language terms.
    """
    def __init__(self, word2id=None):
        """ Init Vocab Instance.
        @param word2id (dict): dictionary mapping words 2 indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad Token
            self.word2id['<unk>'] = 1   # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if word is captured by Vocab.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the Vocab.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in Vocab.
        @returns len (int): number of words in Vocab
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """ Return mapping of index to word.
        @param wid (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to Vocab, if it is previously unseen.
        @param word (str): word to add to Vocab
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, docs):
        """ Convert list of words or list of sentences of words or a list of
        documents of sentences of words into list or list of list of indices.
        @param docs (list[str] or list[list[str]] or list[list[list[str]]]): doc/sent(s) in words
        @return word_ids (list[int] or list[list[int]] or list[list[list[str]]]): doc(s) in indices
        """
        if type(docs[0]) == list:
            if type(docs[0][0]) == list:
                return [[[self[w] for w in s] for s in d] for d in docs]
            else:
                return [[self[w] for w in s] for s in docs]
        else:
            return [self[w] for w in docs]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, docs: List[List[List[str]]], device: torch.device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for
        shorter sentences.

        @param docs (List[List[List[str]]]): list of documents, each of which is a list of sentences (words)
        @param device: device on which to load the tensor, i.e. CPU or GPU

        @returns docs_padded: tensor of (batch_size, max_doc_length, max_sent_length)
        """
        word_ids = self.words2indices(docs)

        max_doc_length = max([len(doc) for doc in word_ids])
        max_sent_length = max([len(sent) for doc in word_ids for sent in doc])

        docs_padded = [[s + [self.word2id['<pad>']]*(max_sent_length-len(s)) for s in d] +
            [[self.word2id['<pad>']]*max_sent_length] * (max_doc_length-len(d)) for d in word_ids]
        
        docs_padded = torch.tensor(docs_padded, dtype=torch.long, device=device)
        return docs_padded

    @staticmethod
    def from_corpus(corpus, freq_cutoff=2):
        """ Given a corpus construct a Vocab Entry.
        @param corpus (list[list[str]]): corpus of text (originally it was list[str])
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab (Vocab): Vocab instance produced from provided corpus
        """
        vocab = Vocab()

        # Added this line, flattening out the document-level list
        corpus_flat = [sent for doc in corpus for sent in doc] # This is now list(str)

        word_freq = Counter(chain(*corpus_flat))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)
        for word in top_k_words:
            vocab.add(word)
        return vocab

    def save(self, file_path):
        """ Save Vocab to file as JSON dump.
        @param file_path (str): file path to vocab file
        """
        json.dump(dict(word2id=self.word2id), open(file_path, 'w'), indent=2)

    @staticmethod
    def load(file_path):
        """ Load vocabulary from JSON dump.
        @param file_path (str): file path to vocab file
        @returns Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r'))
        word2id = entry['word2id']

        return Vocab(word2id)
