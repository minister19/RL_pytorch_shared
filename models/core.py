import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2022-01-26 Shawn: for RL with simple mlp, pure CPU is faster.
device = torch.device("cpu")


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, bias=True, activation=nn.ReLU, output_activation=nn.Identity):
    '''
    Multilayer Perceptron (MLP)
    2021-11-15 Shawn: one can use output_activation=nn.Tanh and act_limit to control action space.
    2021-11-21 Shawn: well I believe this suits symmetric continuous control problem.
    '''
    layers = []
    layers_count = len(sizes) - 1
    for j in range(layers_count):
        act = activation if j < layers_count-1 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1], bias=bias), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class Buffer():
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, *transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)  # [].pop(0) is unefficient.
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def clear(self):
        self.memory.clear()
        self.position = 0


class Benchmark():
    def __init__(self) -> None:
        self.sum = 0
        self.__count = 0
        self.avg = 0

    def plus(self, value):
        self.sum += value
        self.__count += 1
        self.avg = self.sum / self.__count

    def clear(self):
        self.sum = 0
        self.__count = 0
        self.avg = 0
