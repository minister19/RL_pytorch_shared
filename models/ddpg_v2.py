import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from core import mlp


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        net_sizes = [input_size] + list(hidden_size) + [output_size]
        self.net = mlp(net_sizes, True, nn.ReLU, nn.Tanh)

    def forward(self, s):
        a = self.net(s)  # Tensor: [a_dim]
        return a


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        net_sizes = [input_size] + list(hidden_size) + [1]
        self.net = mlp(net_sizes, True, nn.ReLU, nn.Identity)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        q = self.net(x)  # Tensor: [batch_size]
        return q


class Buffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, *transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)  # [].pop(0) is unefficient.
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]

        hidden_size = [256, 256]
        self.actor = Actor(s_dim, hidden_size, a_dim)
        self.actor_target = Actor(s_dim, hidden_size, a_dim)
        self.critic = Critic(s_dim+a_dim, hidden_size)
        self.critic_target = Critic(s_dim+a_dim, hidden_size)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.buffer = Buffer(self.capacity)

    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)  # Tensor: [1, s_dim]
        a0 = self.actor(s0).squeeze(0).detach().numpy()  # Tensor -> ndarray: [a_dim]
        return a0

    def learn(self):
        if len(self.buffer.memory) < self.batch_size:
            return

        samples = random.sample(self.buffer.memory, self.batch_size)

        s0, a0, r1, s1, d = zip(*samples)

        s0 = torch.tensor(np.array(s0), dtype=torch.float)  # [batch_size, s_dim]
        a0 = torch.tensor(np.array(a0), dtype=torch.float)  # [batch_size, a_dim]
        r1 = torch.tensor(np.array(r1), dtype=torch.float).view(self.batch_size, -1)  # [batch_size, 1]
        s1 = torch.tensor(np.array(s1), dtype=torch.float)  # [batch_size, s_dim]
        d = torch.tensor(np.array(d), dtype=torch.float).view(self.batch_size, -1)  # [batch_size, 1]

        def critic_learn():
            a1 = self.actor_target(s1).detach()
            # 2021-12-01 Shawn: done should be considered and learned once.
            y_true = r1 + self.gamma * (1-d) * self.critic_target(s1, a1).detach()

            y_pred = self.critic(s0, a0)

            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            for param in self.critic.parameters():
                param.grad.data.clamp_(-1, 1)
            self.critic_optim.step()

        def actor_learn():
            loss = -torch.mean(self.critic(s0, self.actor(s0)))
            self.actor_optim.zero_grad()
            loss.backward()
            for param in self.actor.parameters():
                param.grad.data.clamp_(-1, 1)
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

env = gym.make('Pendulum-v1')
env.reset()
env.render()

params = {
    'env': env,
    'gamma': 0.99,
    'actor_lr': 0.001,
    'critic_lr': 0.001,
    'tau': 0.02,
    'capacity': 10000,
    'batch_size': 32,
}

agent = Agent(**params)

for episode in range(1000):
    s0 = env.reset()
    done = False
    episode_reward = 0

    for step in range(500):
        env.render()
        a0 = agent.act(s0)
        s1, r1, done, _ = env.step(a0)
        agent.buffer.store(s0, a0, r1, s1, done)  # Ensure all data stored are of type ndarray.

        episode_reward += r1
        s0 = s1

        agent.learn()

        if done:
            print(episode, ': ', episode_reward)
            break

'''
Reference: 
https://spinningup.openai.com/en/latest/algorithms/ddpg.html
https://blog.csdn.net/qq_41871826/article/details/108540108
https://zhuanlan.zhihu.com/p/65931777
'''
