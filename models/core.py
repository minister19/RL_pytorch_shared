import numpy as np
import torch.nn as nn


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
