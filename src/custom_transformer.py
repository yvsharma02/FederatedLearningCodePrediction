import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_attention import MultiHeadedAttention
from attention_head import AttentionHead

class Block(nn.Module):
    def __init__(self, dict_size, context_window_size, embedding_dimensions, head_size):
        super(Block, self).__init__()

        self.network = nn.Sequential(
            MultiHeadedAttention(embedding_dimensions, head_size, context_window_size, 'encoder'),
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
        self.context_window_size = context_window_size
        self.token_embedding = nn.Embedding(dict_size, embedding_dimensions)
        self.position_embedding = nn.Embedding(context_window_size, embedding_dimensions)
        self.decoder = nn.Linear(embedding_dimensions, dict_size)

        self.network = nn.Sequential(
            *[Block(dict_size, context_window_size, embedding_dimensions, head_size) for _ in range(0, block_count)]
        )

    def embed(self, input, spatial = False):
        emb = self.token_embedding(input)
        if (spatial):
            return emb + self.position_embedding(torch.arange(0, self.context_window_size))

        return emb

    # Raw input is a tesnor of (B, W). It should have already mapped tokens to integer.
    def forward(self, raw_input, targets):
        input = self.embed(raw_input, True)
        logits = self.network(input)
        logits = self.decoder(logits)
        if (targets != None):
            logits_1d = logits.view(logits.shape[0] * logits.shape[1], logits.shape[2])
            targets = targets.view(logits_1d.shape[0])
            loss = F.cross_entropy(logits_1d, targets)
        else:
            loss = None
        #print(loss)
        return logits, loss