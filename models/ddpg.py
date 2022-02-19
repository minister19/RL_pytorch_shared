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
        self.net = mlp(net_sizes, True, nn.ReLU, nn.Tanh)
        self.to(device)

    def forward(self, s):
        a = self.net(s)  # Tensor: [[a_dim]]
        return a


class Critic(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        net_sizes = [input_size] + list(hidden_sizes) + [1]
        self.net = mlp(net_sizes, True, nn.ReLU, nn.Identity)  # Q-learning
        self.to(device)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        q = self.net(x)  # Tensor: [batch_size]
        return q


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.s_dim = self.env.observation_space.shape[0]
        self.a_dim = self.env.action_space.shape[0]
        self.a_limit = self.env.action_space.high[0]

        self.actor = Actor(self.s_dim, self.hidden_sizes, self.a_dim)
        self.actor_target = Actor(self.s_dim, self.hidden_sizes, self.a_dim)
        self.critic = Critic(self.s_dim+self.a_dim, self.hidden_sizes)
        self.critic_target = Critic(self.s_dim+self.a_dim, self.hidden_sizes)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.buffer = Buffer(self.capacity)
        self.steps = 0

    def act(self, s0):
        self.steps += 1
        if self.steps < self.start_steps:
            a0 = torch.randn(self.a_dim) * self.a_limit
        else:
            s0 = torch.tensor(s0, dtype=torch.float, device=device)  # Tensor: [s_dim]
            with torch.no_grad():
                a0 = self.actor(s0)
            a0 += torch.randn_like(a0) * self.a_limit * self.action_noise
            a0 = torch.clip(a0, -self.a_limit, self.a_limit)
        return a0.numpy()  # Tensor -> ndarray: [a_dim]

    def learn(self):
        if len(self.buffer.memory) < self.batch_size:
            return

        samples = random.sample(self.buffer.memory, self.batch_size)
        s0, a0, r1, s1, d = zip(*samples)
        s0 = torch.stack(s0)  # [batch_size, s_dim]
        a0 = torch.stack(a0).view(self.batch_size, -1)  # [batch_size, 1]
        r1 = torch.stack(r1).view(self.batch_size, -1)  # [batch_size, 1]
        s1 = torch.stack(s1)  # [batch_size, s_dim]
        d = torch.stack(d).view(self.batch_size, -1)  # [batch_size, 1]

        def critic_learn():
            q = self.critic(s0, a0)

            with torch.no_grad():
                a1 = self.actor_target(s1)
                q_pi_targ = self.critic_target(s1, a1)
                y_true = r1 + self.gamma * (1-d) * q_pi_targ

            loss_fn = nn.MSELoss()
            loss_q = loss_fn(q, y_true)

            self.critic_optim.zero_grad()
            loss_q.backward()
            for param in self.critic.parameters():
                param.grad.data.clamp_(-1, 1)
            self.critic_optim.step()

        def actor_learn():
            pi = self.actor(s0)
            q_pi = self.critic(s0, pi)

            loss_pi = -torch.mean(q_pi)

            self.actor_optim.zero_grad()
            loss_pi.backward()
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


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    env_test = None
    # env.seed(0)
    # np.random.seed(0)
    # random.seed(0)
    # torch.manual_seed(0)

    params = {
        'env': env,
        'env_test': env_test,
        'step_render': False,
        'hidden_sizes': [32, 32],
        'start_steps': 1000,
        'action_noise': 0.1,
        'gamma': 0.99,
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'tau': 0.02,
        'capacity': 10000,
        'batch_size': 64,
    }

    agent = Agent(**params)
    train_score = Benchmark()

    for episode in range(1000):
        s0 = env.reset()
        eps_reward = 0

        for step in range(500):
            if agent.step_render:
                env.render()
            a0 = agent.act(s0)
            s1, r1, d, _ = env.step(a0)

            _s0 = torch.tensor(s0, dtype=torch.float, device=device)
            _a0 = torch.tensor(a0, dtype=torch.float, device=device)
            _r1 = torch.tensor(r1, dtype=torch.float, device=device)
            _s1 = torch.tensor(s1, dtype=torch.float, device=device)
            _d = torch.tensor(d, dtype=torch.float, device=device)
            agent.buffer.store(_s0, _a0, _r1, _s1, _d)

            s0 = s1
            eps_reward += r1

            agent.learn()

            if d:
                train_score.plus(eps_reward)
                print(f'{episode+1}: {step+1} {eps_reward:.2f} {train_score.avg:.2f}')
                break

'''
Reference: 
https://spinningup.openai.com/en/latest/algorithms/ddpg.html
https://zhuanlan.zhihu.com/p/65931777
'''
