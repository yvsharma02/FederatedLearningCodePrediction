import torch
import torch.nn
from custom_transformer import CustomTransformer

torch.manual_seed(1234)

map = {
    'a': 0,
    'b': 1
}

transformer = CustomTransformer(len(map.keys()), 2, 4)

data = [
    [0, 1, 1],
    [1, 0, 1]
]

transformed = transformer.__forward__(torch.tensor(data))
print(transformed.shape)
print(transformed)
# print(
    # transformer.__forward__(
        # torch.tensor(data)
# ))