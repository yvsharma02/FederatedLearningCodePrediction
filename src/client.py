from collections import OrderedDict
import warnings

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import math
import os
from datetime import datetime

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch._dynamo.config.suppress_errors = True

class AttentionHead(nn.Module):

    def __init__(self, emb_dim, head_size, context_window_len, mask):
        super(AttentionHead, self).__init__()
        # These Layers Map (B, W, E) -> (B, W, HEAD_SIZE)

        assert mask == 'encoder' or mask == 'decoder'

        self.key = nn.Linear(emb_dim, head_size, bias=False, device=device)
        self.query = nn.Linear(emb_dim, head_size, bias=False, device=device)
        self.value = nn.Linear(emb_dim, head_size, bias=False, device=device)
        self.mask_type = mask
        self.context_window_len = context_window_len
        self.head_size = head_size

    # Returns a mask of (W, W)
    def get_mask_tensor(self):
        if (self.mask_type == 'encoder'):
            return torch.tril(torch.ones(self.context_window_len, self.context_window_len, device=device))
        elif (self.mask_type == 'decoder'):
            return torch.ones(self.context_window_len, self.context_window_len, device=device)
    
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

class MultiHeadedAttention(nn.Module):

    def __init__(self, emb_dim, num_heads, context_window_len, mask):
        super(MultiHeadedAttention, self).__init__()
        assert emb_dim % num_heads == 0
        self.attention_heads = [AttentionHead(emb_dim, emb_dim // num_heads, context_window_len, mask) for i in range(0, num_heads)]
        

    # Input is (B, W, E)
    def forward(self, input):
        # Each ah returns (B, W, E/num_heads)
        return torch.cat([ah(input) for ah in self.attention_heads], dim= -1)


class Block(nn.Module):
    def __init__(self, context_window_size, embedding_dimensions, num_heads, hidden_layer_multiplier = 4, dropout_rate = 0.3):
        super(Block, self).__init__()

        self.network = nn.Sequential(
            MultiHeadedAttention(embedding_dimensions, num_heads, context_window_size, 'encoder'),
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
        print(raw_input.shape)
        if(raw_input.device != device):
            raw_input = raw_input.to(device)
        input = self.embed(raw_input, True)
        print(input.shape)
        logits = self.network(input)
        logits = self.decoder(logits)
        print(logits.shape)
        if (targets != None):
            if(targets.device != device):
                targets = targets.to(device)
            logits_1d = logits.view(logits.shape[0] * logits.shape[1], logits.shape[2])
            targets = targets.view(logits_1d.shape[0])
            loss = F.cross_entropy(logits_1d, targets)
        else:
            loss = None
        return logits, loss

DELIMITER = "<|endoftext|>"
PADDING = "<|padding|>"
VOCAB_SIZE = 2500

DELIM_ENCODED = 0
PADDING_ENCODED = 1

CONTEXT_LEN = 4
BLOCK_COUNT = 2
EMBED_DIM = 10
NUM_HEADS = 2
LEARNING_RATE = 1e-2
BATCH_COUNT = 2
ITERATIONS = 1

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################
#warnings.filterwarnings("ignore", category=UserWarning)
tokenizer = ByteLevelBPETokenizer("data/tokenizer/vocab.json", "data/tokenizer/merges.txt")

def split(x : list, delim):
    ind = x.index(delim) if delim in x else -1
    if (ind == -1):
        return [x]
    y = split(x[ind + 1:], delim)
    cur = x[:ind]
    return [x[:ind]] + y if len(cur) != 0 else y

#print(split([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4], 1))

def get_padded_longest_sample(encoded : list[int]):
    splits = split(encoded, DELIM_ENCODED)
    splits.sort(key=lambda x: -len(x))
    item = splits[0]
    if (len(splits) > 1):
        splits[0].append(DELIM_ENCODED)
#    print((CONTEXT_LEN - len(splits[0])))
    res = [PADDING_ENCODED] * (CONTEXT_LEN - len(splits[0]))
    res = res + splits[0]
#    print(res)
#    print("Final: ", tokenizer.decode(res))
    return res;

def remove_padding(encoded : list[int]):
    return [x for i, x in enumerate(encoded) if x not in [DELIM_ENCODED, PADDING_ENCODED]]

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, context_len):
        self.data = data.ids
        print(self.data[0:100])
        self.context_len = context_len

    # ABCDE for context len of 1 has 4 examples: (A, B), (B, C), (C, D), (D, E)
    # for context len 2 has examples 3 (AB, C), (BC, D), (CD, E)
    def __len__(self):
        return len(self.data) - self.context_len

    def __getitem__(self, idx):
        return torch.tensor(get_padded_longest_sample(self.data[idx : idx + self.context_len]), device = device), torch.tensor(get_padded_longest_sample(self.data[idx + 1 :idx + self.context_len + 1]), device=device)

def train(transformer, dataset, epochs):
  optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)

  loss_history = []
  history = []
  loss_avg_block_size = 10

  out_dir = "../data/out22M/"
  if not os.path.exists(out_dir):
      os.makedirs(out_dir)

  i = 0
  for current_train_in, current_train_target in dataset:
      start_time = datetime.now()
      print(current_train_in.shape)
      print(current_train_target.shape)
      # Process the current batch
      logits, loss = transformer(current_train_in, current_train_target)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      
      history.append(loss.item())
      print (f"Iteration {i} Loss: {history[-1]}")
      if (len(history) >= loss_avg_block_size):
          loss_history.append(torch.tensor(history).mean().item())
          history = []
          print (f"Iteration {i} Loss: {loss_history[-1]}")
      i += 1

      if (not os.path.exists(out_dir)):
          os.makedirs(out_dir)

      torch.save(transformer.state_dict(), os.path.join(out_dir, "model.pt"))
      end_time = datetime.now()

      total_seconds = (end_time - start_time).total_seconds()

      with open(os.path.join(out_dir, "loss.txt"), "a+") as f:
          f.write(f"Loss: {str(loss)}___________Time: {total_seconds}s\n")

      if (i >= epochs):
          break

  if (len(history) > 0):
      loss_history.append(torch.tensor(history).mean().item())

def test(net, testloader, count):
  """Validate the model on the test set."""
  criterion = torch.nn.CrossEntropyLoss()
  total_loss = 0
  c = 0
  with torch.no_grad():
    for x, y in testloader:
      logits, loss = net(x, y)
      total_loss += loss
      c += 1
      if (c >= count):
        break
  return total_loss / count

def load_data():
    
    train = open("data/CoDesc/fragmented/train_utf8.txt", "r").read()[0:20]
    test = open("data/CoDesc/fragmented/test_utf8.txt", "r").read()[0:20]

    train_enc = tokenizer.encode(train)
    test_enc = tokenizer.encode(test)

    return DataLoader(TextDataset(train_enc, CONTEXT_LEN), BATCH_COUNT, shuffle=True), DataLoader(TextDataset(test_enc, CONTEXT_LEN), BATCH_COUNT, shuffle=True)
  # """Load CIFAR-10 (training and test set)."""
  # trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  # trainset = CIFAR10("./data", train=True, download=True, transform=trf)
  # testset = CIFAR10("./data", train=False, download=True, transform=trf)
  # return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

# #############################################################################
# Federating the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = CustomTransformer(VOCAB_SIZE, CONTEXT_LEN, EMBED_DIM, NUM_HEADS, BLOCK_COUNT)
trainloader, testloader = load_data()

#print(net.state_dict())

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    print("A")
    res = [val.cpu().numpy() for _, val in net.state_dict().items()]
    print(len(res))
    return res

  def set_parameters(self, parameters):
    print("B")
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

  def fit(self, parameters, config):
    print("C")
    self.set_parameters(parameters)
    print("Called Fit!")
    train(net, trainloader, epochs=1)
    print("UnCalled Fit!")
    return self.get_parameters(config={}), len(trainloader.dataset), {}

  def evaluate(self, parameters, config):
    print("D")
    self.set_parameters(parameters)
    total_loss = 0
    c = 0
    for x,y in testloader:
        print("_X_X_X_")
        print(x.shape)
        print(y.shape)
        logits, loss = net(x, y)
        total_loss += loss
        c += 1

    return float(loss), len(testloader.dataset), {"loss": float(total_loss / c)}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=FlowerClient())