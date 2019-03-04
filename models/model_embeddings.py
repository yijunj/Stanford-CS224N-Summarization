
"""
Word-level embedding
"""

import torch.nn as nn

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        self.vocab = vocab

        src_pad_token_idx = vocab['<pad>']

        self.source = nn.Embedding( len(vocab), self.embed_size, padding_idx=src_pad_token_idx )
