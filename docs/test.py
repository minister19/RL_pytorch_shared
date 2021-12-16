# %%
from torch.distributions import Categorical
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
x = list([256, 256])
print(x)

x = np.array([1])
print(x)

# %%
t = torch.tensor([[1, 2], [3, 4]])
x = torch.gather(t, 1, torch.tensor([[0, 0, 1]]))
print(x)
x = torch.gather(t, 1, torch.tensor([[0], [1]]))
print(x)
x = torch.gather(t, 1, torch.tensor([[0, 1, 0], [1, 1, 0]]))
print(x)

# %%
x = torch.nn.Module()
params = itertools.chain(x.parameters(), x.parameters())
print(type(params))

# %%
x = random.uniform(0, 2.5)
print(x)

# %%
x = torch.tensor([1.1, 1.2])
x = x.squeeze(0).numpy()
print(x)

# %%
input = torch.randn(2, 3)
print(input)

x = F.softmax(input, 0)
print(x)
x = F.softmax(input, 1)
print(x)

m = nn.Softmax(dim=1)
output = m(input)
print(output)

# %%
src = torch.arange(1, 11).reshape((2, 5))
print(src)

index = torch.tensor([[0, 1, 2, 0]])
x = torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
print(x)

index = torch.tensor([[0, 1, 2, 0]])
x = torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
print(x)

index = torch.tensor([[0, 1, 2], [0, 1, 4]])
x = torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
print(x)

# %%
src = torch.rand(2, 5)
print(src)
index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]])
print(index)
x = torch.zeros(3, 5).scatter_(0, index, src)
print(x)
x = torch.zeros(3, 5).scatter_(1, index, src)
print(x)

# %%
index = torch.tensor([0, 1, 1, 1, 0])
index = index.view(5, -1)
print(index)
x = torch.zeros(10, 2).scatter_(1, index, 1)
print(x)

# %%
src = torch.randn(8, 2)
print(src)
logits = torch.log(F.softmax(src, dim=1))
print(logits)
# Sample hard categorical using "Straight-through" trick:
x = F.gumbel_softmax(logits, tau=1, hard=True, dim=1)
print(x)
x = torch.argmax(logits, dim=1, keepdim=True)
print(x)

# %%
x = random.randint(0, 1)
print(x)

# %%
p = np.array([0.1, 0.2, 0.4, 0.3])
logp = np.log(p)
entropy1 = np.sum(-p*logp)
print(entropy1)

p_tensor = torch.Tensor([0.1, 0.2, 0.4, 0.3])
entropy2 = Categorical(probs=p_tensor).entropy()
print(entropy2)
