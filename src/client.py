from collections import OrderedDict

import flwr as fl
import torch
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader

import os
from datetime import datetime

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from custom_transformer import CustomTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)
torch._dynamo.config.suppress_errors = True

DELIMITER = "<|endoftext|>"
PADDING = "<|padding|>"
VOCAB_SIZE = 2500

DELIM_ENCODED = 0
PADDING_ENCODED = 1

CONTEXT_LEN = 64
BLOCK_COUNT = 2
EMBED_DIM = 256
NUM_HEADS = 16
LEARNING_RATE = 1e-2
BATCH_COUNT = 32
ITERATIONS = 1000

tokenizer = ByteLevelBPETokenizer("data/tokenizer/vocab.json", "data/tokenizer/merges.txt")

def split(x : list, delim):
    ind = x.index(delim) if delim in x else -1
    if (ind == -1):
        return [x]
    y = split(x[ind + 1:], delim)
    cur = x[:ind]
    return [x[:ind]] + y if len(cur) != 0 else y


def get_padded_longest_sample(encoded : list[int]):
    splits = split(encoded, DELIM_ENCODED)
    splits.sort(key=lambda x: -len(x))
    item = splits[0]
    if (len(splits) > 1):
        splits[0].append(DELIM_ENCODED)
    res = [PADDING_ENCODED] * (CONTEXT_LEN - len(splits[0]))
    res = res + splits[0]
    return res

def remove_padding(encoded : list[int]):
    return [x for i, x in enumerate(encoded) if x not in [DELIM_ENCODED, PADDING_ENCODED]]

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, context_len):
        self.data = data.ids
        self.context_len = context_len

    # ABCDE for context len of 1 has 4 examples: (A, B), (B, C), (C, D), (D, E)
    # for context len 2 has examples 3 (AB, C), (BC, D), (CD, E)
    def __len__(self):
        return len(self.data) - self.context_len

    def __getitem__(self, idx):
        return torch.tensor(get_padded_longest_sample(self.data[idx : idx + self.context_len]), device = device), torch.tensor(get_padded_longest_sample(self.data[idx + 1 :idx + self.context_len + 1]), device=device)

def train(transformer, dataset, samples):
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

      if (i >= samples):
          break

  if (len(history) > 0):
      loss_history.append(torch.tensor(history).mean().item())

def test(net, testloader, count):
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
    
    train = open("data/CoDesc/fragmented/train_utf8.txt", "r").read()
    test = open("data/CoDesc/fragmented/test_utf8.txt", "r").read()

    train_enc = tokenizer.encode(train)
    test_enc = tokenizer.encode(test)

    return DataLoader(TextDataset(train_enc, CONTEXT_LEN), BATCH_COUNT, shuffle=True), DataLoader(TextDataset(test_enc, CONTEXT_LEN), BATCH_COUNT, shuffle=True)

net = CustomTransformer(VOCAB_SIZE, CONTEXT_LEN, EMBED_DIM, NUM_HEADS, BLOCK_COUNT)
trainloader, testloader = load_data()

class FlowerClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    res = [val.cpu().numpy() for _, val in net.state_dict().items()]
    return res

  def set_parameters(self, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

  def fit(self, parameters, config):
    self.set_parameters(parameters)
    train(net, trainloader, samples=len(trainloader.dataset))
    return self.get_parameters(config={}), len(trainloader.dataset), {}

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    total_loss = 0
    c = 0
    for x,y in testloader:
        logits, loss = net(x, y)
        total_loss += loss
        c += 1
     
    return float(loss), len(testloader.dataset), {"loss": float(total_loss / c)}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=FlowerClient())