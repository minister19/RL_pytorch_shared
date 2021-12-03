import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from core import mlp, Buffer


class Actor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        net_sizes = [input_size] + list(hidden_sizes) + [output_size]
        self.net = mlp(net_sizes, True, nn.ReLU, nn.Identity)  # Q-learning

    def forward(self, s):
        a = self.net(s)  # Tensor: [a_dim]
        return a


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.s_dim = self.env.observation_space.shape[0]
        self.a_dim = self.env.action_space.n

        hidden_sizes = [64, 64]
        self.q_net = Actor(self.s_dim, hidden_sizes, self.a_dim)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        self.buffer = Buffer(self.capacity)
        self.steps = 0

    def act(self, s0):
        self.steps += 1
        epsi = self.epsi_low + (self.epsi_high-self.epsi_low) * (math.exp(-1.0 * self.steps/self.decay))
        if random.random() < epsi:
            a0 = random.randrange(self.a_dim)
        else:
            s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
            a0 = self.q_net(s0).detach()
            a0 = torch.argmax(a0).item()
        return a0  # torch.long

    def learn(self):
        if (len(self.buffer.memory)) < self.batch_size:
            return

        samples = random.sample(self.buffer.memory, self.batch_size)
        s0, a0, r1, s1, d = zip(*samples)
        s0 = torch.tensor(np.array(s0), dtype=torch.float)  # [batch_size, s_dim]
        a0 = torch.tensor(np.array(a0), dtype=torch.long).view(self.batch_size, -1)  # [batch_size, 1]
        r1 = torch.tensor(np.array(r1), dtype=torch.float).view(self.batch_size, -1)  # [batch_size, 1]
        s1 = torch.tensor(np.array(s1), dtype=torch.float)  # [batch_size, s_dim]
        d = torch.tensor(np.array(d), dtype=torch.float).view(self.batch_size, -1)  # [batch_size, 1]

        def q_net_learn():
            a1 = self.q_net(s1).detach()  # [batch_size, a_dim]
            a1_values, a1_indices = torch.max(a1, dim=1)  # [batch_size]
            y_true = r1 + self.gamma * (1-d) * a1_values.view(self.batch_size, -1)

            y_pred = self.q_net(s0).gather(dim=1, index=a0)

            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.q_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        q_net_learn()


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

env = gym.make('CartPole-v1')

params = {
    'env': env,
    'gamma': 0.8,
    'epsi_high': 0.9,
    'epsi_low': 0.05,
    'decay': 200,
    'lr': 0.001,
    'capacity': 10000,
    'batch_size': 64,
}
agent = Agent(**params)

eps_reward_sum = 0

for episode in range(1000):
    s0 = env.reset()
    done = False
    eps_reward = 0

    for step in range(500):
        env.render()
        a0 = agent.act(s0)
        s1, r1, done, _ = env.step(a0)
        # 2021-12-02 Shawn: redefine reward for better control target and convergence.
        r1 = -1 * (abs(s1[0])/2.4 + abs(s1[2])/0.209)
        agent.buffer.store(s0, a0, r1, s1, done)

        eps_reward += r1
        s0 = s1

        agent.learn()

        if done:
            eps_reward_sum += eps_reward
            eps_reward_avg = eps_reward_sum / (episode+1)
            print(f'{episode+1}: {step+1} {eps_reward:.2f} {eps_reward_avg:.2f}')
            break

'''
Reference:
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
https://zhuanlan.zhihu.com/p/40226021
'''
