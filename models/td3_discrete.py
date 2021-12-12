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

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        q = self.net(x)  # Tensor: [batch_size]
        return q


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.s_dim = self.env.observation_space.shape[0]
        self.a_dim = self.env.action_space.n

        hidden_sizes = [8, 8]
        self.actor = Actor(self.s_dim, hidden_sizes, self.a_dim)
        self.actor_target = Actor(self.s_dim, hidden_sizes, self.a_dim)
        self.critic1 = Critic(self.s_dim+self.a_dim, hidden_sizes)
        self.critic1_target = Critic(self.s_dim+self.a_dim, hidden_sizes)
        self.critic2 = Critic(self.s_dim+self.a_dim, hidden_sizes)
        self.critic2_target = Critic(self.s_dim+self.a_dim, hidden_sizes)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_params = itertools.chain(self.critic1.parameters(), self.critic2.parameters())
        self.critic_optim = optim.Adam(self.critic_params, lr=self.critic_lr)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.buffer = Buffer(self.capacity)
        self.steps = 0

    def act(self, s0):
        self.steps += 1
        if self.steps < self.start_steps:
            a0 = torch.randint(self.a_dim, (1,))
        else:
            s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)  # Tensor: [1, s_dim]
            with torch.no_grad():
                a0 = self.actor(s0)
            a0 = torch.argmax(a0)
        return a0.item()

    def learn(self):
        if len(self.buffer.memory) < self.batch_size:
            return

        samples = random.sample(self.buffer.memory, self.batch_size)
        s0, a0, r1, s1, d = zip(*samples)
        s0 = torch.tensor(np.array(s0), dtype=torch.float)  # [batch_size, s_dim]
        a0 = torch.tensor(np.array(a0), dtype=torch.long).view(self.batch_size, -1)  # [batch_size, 1]
        a0_one_hot = torch.zeros(self.batch_size, self.a_dim).scatter_(1, a0, 1)  # [batch_size, a_dim]
        r1 = torch.tensor(np.array(r1), dtype=torch.float).view(self.batch_size, -1)  # [batch_size, 1]
        s1 = torch.tensor(np.array(s1), dtype=torch.float)  # [batch_size, s_dim]
        d = torch.tensor(np.array(d), dtype=torch.float).view(self.batch_size, -1)  # [batch_size, 1]

        def critic_learn():
            q1 = self.critic1(s0, a0_one_hot)
            q2 = self.critic2(s0, a0_one_hot)

            with torch.no_grad():
                a1 = self.actor_target(s1)
                # Target policy smoothing disabled
                a1_values, a1_indices = torch.max(a1, dim=1, keepdim=True)  # [batch_size, a_dim]
                a1_one_hot = torch.zeros_like(a1).scatter_(1, a1_indices, 1)

                # Target Q-values
                q1_pi_targ = self.critic1_target(s1, a1_one_hot)
                q2_pi_targ = self.critic2_target(s1, a1_one_hot)
                # v1: best critic
                # q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                # v2: random critic
                if random.randint(0, 1) == 0:
                    q_pi_targ = q1_pi_targ
                else:
                    q_pi_targ = q2_pi_targ
                y_true = torch.zeros_like(q1)
                for i in range(self.batch_size):
                    y_true[i] = r1[i] + self.gamma * (1-d[i]) * q_pi_targ[i]

            # MSE loss against Bellman backup
            loss_fn = nn.MSELoss()
            loss_q1 = loss_fn(q1, y_true)
            loss_q2 = loss_fn(q2, y_true)
            loss_q = loss_q1 + loss_q2

            self.critic_optim.zero_grad()
            loss_q.backward()
            for param in self.critic_params:
                param.grad.data.clamp_(-1, 1)
            self.critic_optim.step()

        def actor_learn():
            pi = self.actor(s0)
            logits = F.log_softmax(pi, dim=1)
            differentiable_pi = F.gumbel_softmax(logits, hard=True)
            index = torch.argmax(pi, dim=1, keepdim=True)
            pi_one_hot = torch.zeros_like(pi).scatter_(1, index, 1)  # [batch_size, a_dim]
            q1_pi = self.critic1(s0, (differentiable_pi + pi_one_hot)/2)
            q2_pi = self.critic2(s0, (differentiable_pi + pi_one_hot)/2)
            # v1: always critic1
            # q_pi = q1_pi
            # v2: best critic
            q_pi = torch.min(q1_pi, q2_pi)

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
        # Delayed Policy Updates disabled
        actor_learn()
        soft_update(self.critic1_target, self.critic1, self.tau)
        soft_update(self.critic2_target, self.critic2, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)


env = gym.make('CartPole-v0')
# env.seed(0)
# np.random.seed(0)
# random.seed(0)
# torch.manual_seed(0)

params = {
    'env': env,
    'step_render': False,
    'start_steps': 1000,
    'gamma': 0.5,
    'actor_lr': 0.001,
    'critic_lr': 0.001,
    'tau': 0.02,
    'capacity': 10000,
    'batch_size': 64,
}

agent = Agent(**params)

eps_reward_sum = 0

for episode in range(1000):
    s0 = env.reset()
    eps_reward = 0

    for step in range(500):
        if agent.step_render:
            env.render()
        a0 = agent.act(s0)
        s1, r1, done, _ = env.step(a0)
        r1 = -1 * (abs(s1[2])/0.209)
        # r1 = -1 * (abs(s1[0])/2.4 + abs(s1[2])/0.209)
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
https://spinningup.openai.com/en/latest/algorithms/td3.html
'''
