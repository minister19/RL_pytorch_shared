import numpy as np
import torch

x = list([256, 256])
print(x)

t = torch.tensor([[1, 2], [3, 4]])
x = torch.gather(t, 1, torch.tensor([[0], [1]]))
print(x)
x = torch.gather(t, 1, torch.tensor([[0, 1, 0], [1, 1, 0]]))
print(x)
