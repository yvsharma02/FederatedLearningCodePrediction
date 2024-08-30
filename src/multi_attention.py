from attention_head import AttentionHead
import torch
import torch.nn as nn

class MultiHeadedAttention:

    def __init__(self, emb_dim, num_heads, context_window_len, mask):

        assert emb_dim % num_heads == 0
        self.attention_heads = [AttentionHead(emb_dim, num_heads, context_window_len, mask) for i in range(0, emb_dim / num_heads)]

    # Input is (B, W, E)
    def forward(self, input):
        # Each ah returns (B, W, E/num_heads)
        return torch.cat([ah(input) for ah in self.attention_heads], dim= -1)
