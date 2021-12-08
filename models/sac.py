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
from torch.distributions.normal import Normal
from core import mlp, Buffer


class Actor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, log_std_min=-20, log_std_max=2):
        super().__init__()
        net_sizes = [input_size] + list(hidden_sizes)
        self.net = mlp(net_sizes, True, nn.ReLU, nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        return pi_action, logp_pi


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
        self.a_dim = self.env.action_space.shape[0]

        hidden_sizes = [64, 64]
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

    def act(self, s0, eps_step):
        if eps_step < self.init_wander:
            a0 = random.uniform(-self.a_dim, self.a_dim)
            a0 = np.array([a0])
        else:
            s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)  # Tensor: [1, s_dim]
            with torch.no_grad():
                a0, _ = self.actor(s0, deterministic=False, with_logprob=False)
                a0 = a0.squeeze(0).numpy()  # Tensor -> ndarray: [a_dim]
        return a0

    def learn(self, eps_step):
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
            q1 = self.critic1(s0, a0)
            q2 = self.critic2(s0, a0)

            with torch.no_grad():
                a1, logp_a1 = self.actor(s1)
                logp_a1_viewed = logp_a1.view(self.batch_size, -1)
                q1_pi_targ = self.critic1_target(s1, a1)
                q2_pi_targ = self.critic2_target(s1, a1)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                y_true = r1 + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a1_viewed)

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
            pi, logp_pi = self.actor(s0)
            logp_pi_v2 = logp_pi.view(self.batch_size, -1)
            q1_pi = self.critic1(s0, pi)
            q2_pi = self.critic2(s0, pi)
            q_pi = torch.min(q1_pi, q2_pi)

            # Entropy-regularized policy loss
            loss = torch.mean(self.alpha * logp_pi_v2 - q_pi)

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
        soft_update(self.critic1_target, self.critic1, self.tau)
        soft_update(self.critic2_target, self.critic2, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)


env = gym.make('Pendulum-v1')
# env.seed(0)
# np.random.seed(0)
# random.seed(0)
# torch.manual_seed(0)

params = {
    'env': env,
    'init_wander': 200,
    'alpha': 0.2,
    'gamma': 0.99,
    'actor_lr': 0.005,
    'critic_lr': 0.005,
    'tau': 0.01,
    'capacity': 10000,
    'batch_size': 64,
}

agent = Agent(**params)

eps_step = 0

for episode in range(1000):
    s0 = env.reset()
    eps_reward = 0

    for step in range(500):
        env.render()
        a0 = agent.act(s0, eps_step)
        s1, r1, done, _ = env.step(a0)
        agent.buffer.store(s0, a0, r1, s1, done)

        eps_step += 1
        eps_reward += r1
        s0 = s1

        agent.learn(eps_step)

        if done:
            print(f'{episode+1}: {step+1} {eps_reward:.2f}')
            break

'''
Reference:
https://spinningup.openai.com/en/latest/algorithms/sac.html
https://arxiv.org/pdf/1812.05905.pdf
'''
