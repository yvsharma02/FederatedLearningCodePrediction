import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_attention import MultiHeadedAttention
from attention_head import AttentionHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Block(nn.Module):
    def __init__(self, context_window_size, embedding_dimensions, num_heads, hidden_layer_multiplier = 4, dropout_rate = 0.3):
        super(Block, self).__init__()

        self.network = nn.Sequential(
            MultiHeadedAttention(embedding_dimensions, num_heads, context_window_size, 'decoder'),
            nn.Linear(embedding_dimensions, hidden_layer_multiplier * embedding_dimensions, device=device),
            nn.ReLU(),
            nn.Linear(embedding_dimensions * hidden_layer_multiplier, embedding_dimensions, device=device),
            nn.LayerNorm(embedding_dimensions, device=device),
            nn.Dropout(dropout_rate)
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
    def __init__(self, dict_size, context_window_size, embedding_dimensions, num_heads, block_count):
        super(CustomTransformer, self).__init__()
        self.context_window_size = context_window_size
        self.token_embedding = nn.Embedding(dict_size, embedding_dimensions, device=device)
        self.position_embedding = nn.Embedding(context_window_size, embedding_dimensions, device=device)
        self.decoder = nn.Linear(embedding_dimensions, dict_size, device=device)

        self.network = nn.Sequential(
            *[Block(context_window_size, embedding_dimensions, num_heads) for _ in range(0, block_count)]
        )

    def embed(self, input, spatial = False):
        emb = self.token_embedding(input)
        if (spatial):
            return emb + self.position_embedding(torch.arange(0, self.context_window_size, device=device))

        return emb

    # Raw input is a tesnor of (B, W). On CPU. It should have already mapped tokens to integer.
    def forward(self, raw_input, targets):
        if(raw_input.device != device):
            raw_input = raw_input.to(device)
        input = self.embed(raw_input, True)
        logits = self.network(input)
        logits = self.decoder(logits)
        if (targets != None):
            if(targets.device != device):
                targets = targets.to(device)
            logits_1d = logits.view(logits.shape[0] * logits.shape[1], logits.shape[2])
            targets = targets.view(logits_1d.shape[0])
            loss = F.cross_entropy(logits_1d, targets)
        else:
            loss = None
        return logits, loss