import torch
import torch.nn as nn

from multi_attention import MultiHeadedAttention
from attention_head import AttentionHead

class Block(nn.Module):


    def __init__(self, dict_size, context_window_size, embedding_dimensions, head_size):
        super(CustomTransformer, self).__init__()
        self.token_embedding = nn.Embedding(dict_size, embedding_dimensions)
        self.position_embedding = nn.Embedding(context_window_size, embedding_dimensions)

        self.network = nn.Sequential(
            MultiHeadedAttention(4 * embedding_dimensions, head_size, context_window_size, 'encoder'),
            nn.Linear(embedding_dimensions, 4 * embedding_dimensions),
            nn.ReLU(),
            nn.Linear(embedding_dimensions * 4, embedding_dimensions),
            nn.LayerNorm(embedding_dimensions),
            nn.Dropout(0.1)
        )

    # Raw input is a tesnor of (B, W). It should have already mapped tokens to integer.
    def forward(self, raw_input):
        return self.network(raw_input)

class CustomTransformer(nn.Module):

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
    def __init__(self, dict_size, context_window_size, embedding_dimensions, head_size, block_count):
        super(CustomTransformer, self).__init__()
        self.token_embedding = nn.Embedding(dict_size, embedding_dimensions)
        self.position_embedding = nn.Embedding(context_window_size, embedding_dimensions)

        self.network = nn.Sequential(
            *[Block(self, dict_size, context_window_size, embedding_dimensions, head_size, block_count) for _ in range(0, block_count)]
        )

    def embed(self, input):
        return self.token_embedding(input) + self.position_embedding(input)

    # Raw input is a tesnor of (B, W). It should have already mapped tokens to integer.
    def forward(self, raw_input):

        input = self.embed(raw_input)

        return input