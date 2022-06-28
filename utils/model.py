import torch.nn as nn

from utils.constants import EMBED_DIMENSION, EMBED_MAX_NORM


class CBOW_Model(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()


    def forward(self, inputs_):

        return x


class SkipGram_Model(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()


    def forward(self, inputs_):

        return x

class RNN_classif(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(RNN_classif, self).__init__()

        #
        # We wish to develop a RNN that, from the text, produces positive or negative (1 / 0)
        #

    def forward(self, text): # text = [len, batch_size]

        return ...
