import gym
import random
import itertools
import numpy as np
import torch

x = list([256, 256])
print(x)

t = torch.tensor([[1, 2], [3, 4]])
x = torch.gather(t, 1, torch.tensor([[0], [1]]))
print(x)
x = torch.gather(t, 1, torch.tensor([[0, 1, 0], [1, 1, 0]]))
print(x)

x = torch.nn.Module()
params = itertools.chain(x.parameters(), x.parameters())
print(type(params))

x = random.uniform(0, 2.5)
print(x)

x = torch.tensor([1.1, 1.2])
x = x.squeeze(0).detach().numpy()
print(x)

x = np.array([1])
print(x)

env = gym.make('HalfCheetah-v2')
print(env)
