import torch
import torch.nn as nn

class CustomTransformer():

    # Input to the Transformer will be a matrix of size (B, W)
    # B is the Batch Size.
    # W is the Window Size (context_window_size)
    # Example:
    # [a, b, c]
    # [d, e, f]
    #
    # [a, b, c] is an input example. (context_len = W = 3)
    # There are two batches [a, b, c] and [d, e, f] (B = 2)
    # a, b, c should be integers (each representing one possible token). a, b, c should belong in [0, dict_size)
    def __init__(self, dict_size, context_window_size, embedding_dimensions):

        self.token_embedding = nn.Embedding(dict_size, embedding_dimensions)
        self.position_embedding = nn.Embedding(context_window_size, embedding_dimensions)

        pass

    def embed(self, input):
        return self.token_embedding(input) + self.position_embedding(input)

    def __forward__(self, input):

        input = self.embed(input)

        return input