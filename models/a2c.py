import itertools
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
        self.net = mlp(net_sizes, True, nn.ReLU, nn.Tanh)

    def forward(self, s):
        a = self.net(s)  # Tensor: [[a_dim]]
        return a


class Critic(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        net_sizes = [input_size] + list(hidden_sizes) + [1]
        self.net = mlp(net_sizes, True, nn.ReLU, nn.Identity)  # Q-learning

    def forward(self, s):
        v = self.net(s)
        return v


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.s_dim = self.env.observation_space.shape[0]
        self.a_dim = self.env.action_space.n

        hidden_sizes = [8, 8]
        self.actor = Actor(self.s_dim, hidden_sizes, self.a_dim)
        self.critic = Critic(self.s_dim, hidden_sizes)

        self.params = itertools.chain(self.actor.parameters(), self.critic.parameters())
        self.optim = optim.Adam(self.params, lr=self.lr)

        self.buffer = Buffer(self.capacity)
        self.entropy = 0

    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        probs = self.actor(s0)
        probs = torch.softmax(probs, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        a0 = dist.sample()
        log_prob = dist.log_prob(a0)
        entropy = dist.entropy()
        self.entropy += entropy

        v_current = self.critic(s0).squeeze(0)
        return a0.item(), log_prob, v_current

    def learn(self, s1):
        s1 = torch.tensor(s1, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            v_next = self.critic(s1)

        log_prob, v_current, r1, d = zip(*self.buffer.memory)
        log_prob = torch.stack(log_prob)
        v_current = torch.stack(v_current)

        # Calculate part1 bottom up (from final to init)
        part1 = torch.zeros((self.buffer.position, 1))
        for i in range(self.buffer.position):
            index = -1-i
            v_next = r1[index] + (1-d[index])*self.gamma*v_next
            part1[index] = v_next

        def advantage_learn():
            advantage = part1 - v_current
            critic_loss = 0.5 * advantage.pow(2).mean()
            actor_loss = torch.mean(-log_prob*advantage.detach())
            if self.enable_entropy:
                loss = critic_loss + actor_loss + self.alpha * self.entropy
            else:
                loss = critic_loss + actor_loss

            self.optim.zero_grad()
            loss.backward()
            for param in self.params:
                param.grad.data.clamp_(-1, 1)
            self.optim.step()

        advantage_learn()


env = gym.make('CartPole-v1')
# env.seed(0)
# np.random.seed(0)
# random.seed(0)
# torch.manual_seed(0)

params = {
    'env': env,
    'step_render': False,
    'enable_entropy': True,
    'alpha': 0.001,
    'gamma': 0.999,
    'lr': 0.005,
    'max_steps': 500,
    'capacity': 1000,
}

agent = Agent(**params)

eps_reward_sum = 0

for episode in range(10000):
    s0 = env.reset()
    eps_reward = 0

    for step in range(1000):
        if agent.step_render:
            env.render()
        a0, log_prob, v_current = agent.act(s0)
        s1, r1, done, _ = env.step(a0)
        r1 = 1-1 * (abs(s1[2])/0.209)
        # r1 = 2-1 * (abs(s1[0])/2.4 + abs(s1[2])/0.209)
        agent.buffer.store(log_prob, v_current, r1, done)

        eps_reward += r1
        s0 = s1

        if done or ((step+1) % agent.max_steps == 0):
            agent.learn(s1)
            agent.buffer.clear()
            agent.entropy = 0

        if done:
            eps_reward_sum += eps_reward
            eps_reward_avg = eps_reward_sum / (episode+1)
            print(f'{episode+1}: {step+1} {eps_reward:.2f} {eps_reward_avg:.2f}')
            break

'''
Reference: 
https://medium.com/deeplearningmadeeasy/advantage-actor-critic-a2c-implementation-944e98616b
https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
https://www.freecodecamp.org/news/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d/
https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752
'''
