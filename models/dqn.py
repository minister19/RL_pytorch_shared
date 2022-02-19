import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from core import device, mlp, Buffer, Benchmark


class Actor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        net_sizes = [input_size] + list(hidden_sizes) + [output_size]
        self.net = mlp(net_sizes, False, nn.ReLU, nn.Identity)  # Q-learning
        self.to(device)

    def forward(self, s):
        a = self.net(s)  # Tensor: [[a_dim]]
        return a


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.s_dim = self.env.observation_space.shape[0]
        self.a_dim = self.env.action_space.n

        self.q_net = Actor(self.s_dim, self.hidden_sizes, self.a_dim)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        self.buffer = Buffer(self.capacity)
        self.steps = 0

    def act(self, s0):
        self.steps += 1
        epsilon = self.epsi_low + (self.epsi_high-self.epsi_low) * (math.exp(-1.0 * self.steps/self.decay))
        if random.random() < epsilon:
            a0 = torch.randint(self.a_dim, (1,))
        else:
            s0 = torch.tensor(s0, dtype=torch.float, device=device)
            with torch.no_grad():
                a0 = self.q_net(s0)
            a0 = torch.argmax(a0)
        return a0.item()

    def learn(self):
        if (len(self.buffer.memory)) < self.batch_size:
            return

        samples = random.sample(self.buffer.memory, self.batch_size)
        s0, a0, a0_one_hot, r1, s1, d = zip(*samples)
        s0 = torch.stack(s0)  # [batch_size, s_dim]
        a0 = torch.stack(a0).view(self.batch_size, -1)  # [batch_size, 1]
        a0_one_hot = torch.stack(a0_one_hot).view(self.batch_size, -1)  # [batch_size, a_dim]
        r1 = torch.stack(r1).view(self.batch_size, -1)  # [batch_size, 1]
        s1 = torch.stack(s1)  # [batch_size, s_dim]
        d = torch.stack(d).view(self.batch_size, -1)  # [batch_size, 1]

        def q_net_learn():
            q = self.q_net(s0).gather(dim=1, index=a0)

            with torch.no_grad():
                a1 = self.q_net(s1)  # [batch_size, a_dim]
                a1_values, a1_indices = torch.max(a1, dim=1, keepdim=True)  # [batch_size]
                y = r1 + self.gamma * (1-d) * a1_values

            loss_fn = nn.MSELoss()
            loss_q = loss_fn(q, y)

            self.optimizer.zero_grad()
            loss_q.backward()
            for param in self.q_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        q_net_learn()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env_test = None
    # env.seed(0)
    # np.random.seed(0)
    # random.seed(0)
    # torch.manual_seed(0)
    params = {
        'env': env,
        'env_test': env_test,
        'step_render': False,
        'hidden_sizes': [8, 8],
        'epsi_high': 0.9,
        'epsi_low': 0.05,
        'decay': 200,
        'gamma': 0.5,
        'lr': 0.001,
        'capacity': 10000,
        'batch_size': 64,
    }
    agent = Agent(**params)
    train_score = Benchmark()

    for episode in range(1000):
        s0 = env.reset()
        eps_reward = 0

        for step in range(1000):
            if agent.step_render:
                env.render()
            a0 = agent.act(s0)
            s1, r1, d, _ = env.step(a0)
            # 2021-12-02 Shawn: redefine reward for better control target and convergence.
            r1 = -1 * (abs(s1[2])/0.209)
            # r1 = -1 * (abs(s1[0])/2.4 + abs(s1[2])/0.209)

            _s0 = torch.tensor(s0, dtype=torch.float, device=device)
            _a0 = torch.tensor(a0, dtype=torch.long, device=device)
            _a0_one_hot = torch.zeros(agent.a_dim, device=device).scatter_(0, _a0, 1)
            _r1 = torch.tensor(r1, dtype=torch.float, device=device)
            _s1 = torch.tensor(s1, dtype=torch.float, device=device)
            _d = torch.tensor(d, dtype=torch.float, device=device)
            agent.buffer.store(_s0, _a0, _a0_one_hot, _r1, _s1, _d)

            s0 = s1
            eps_reward += r1

            agent.learn()

            if d:
                train_score.plus(eps_reward)
                print(f'{episode+1}: {step+1} {eps_reward:.2f} {train_score.avg:.2f}')
                break
'''
Reference:
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
https://zhuanlan.zhihu.com/p/40226021
'''
