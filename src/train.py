import torch
import torch.nn
from custom_transformer import CustomTransformer
from attention_head import AttentionHead
from multi_attention import MultiHeadedAttention


# t = torch.rand(2, 3, 4)
# print(t.mean(2))

torch.manual_seed(1234)

map = {
    'a': 0,
    'b': 1
}

x = torch.rand((3, 2, 10), dtype=torch.float32)

mha = MultiHeadedAttention(10, 2, 2, 'encoder')
print(x)
y = mha(x)
print(y)
# transformer = CustomTransformer(len(map.keys()), 2, 4)

# data = [
#     [0, 1, 1, 1],
#     [1, 0, 1, 1]
# ]

# ah = AttentionHead(4, 3, 2, 'encoder')
# print(ah.forward(torch.tensor(data, dtype=torch.float32)))

# # transformed = transformer.__forward__(torch.tensor(data))
# # print(transformed.shape)
# # print(transformed)
# # print(
#     # transformer.__forward__(
#         # torch.tensor(data)
# # ))