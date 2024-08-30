import torch
import torch.nn as nn
import torch.functional as F
import math

class AttentionHead(nn.Module):

    def __init__(self, emb_dim, head_size, context_window_len, mask):
#        super(self).__init__()
        # These Layers Map (B, W, E) -> (B, W, HEAD_SIZE)

        assert mask == 'encoder' or mask == 'decoder'

        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.mask_type = mask
        self.context_window_len = context_window_len
        self.head_size = head_size

    # Returns a mask of (W, W)
    def get_mask_tensor(self):
        if (self.mask_type == 'encoder'):
            return torch.tril(torch.ones(self.context_window_len, self.context_window_len))
        elif (self.mask_type == 'decoder'):
            return torch.ones(self.context_window_len, self.context_window_len)
    
    # Input is of shape (B, W, E) where E is embedding dimensions.
    # Output is of shape (B, W, E)
    def forward(self, input):
        k = self.key(input) # Convert (B, W1, E) -> (B, W1, HEAD_SIZE)
        q = self.query(input) # Convert (B, W2, E) -> (B, W2, HEAD_SIZE) (W1 == W2 == W3)
        v = self.value(input) # (B, W3, E) -> (B, W3, HEAD_SIZE)
        match = q @ k.transpose(-2, -1) # Produce Matrix (B, W1, W2)
        mask = self.get_mask_tensor()
        match = match.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(match, dim=-1) / math.sqrt(self.head_size) # Still (B, W1, W2)

        res = attention @ v # (B, W1, W2) @ (B, W3, HEAD_SIZE) -> (B, W1=W3, HEAD_SIZE)
        return res