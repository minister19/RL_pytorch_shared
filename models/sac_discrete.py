import itertools
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from core import device, mlp, Buffer, Benchmark


class Actor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, log_std_min=-2, log_std_max=2):
        super().__init__()
        net_sizes = [input_size] + list(hidden_sizes)
        self.net = mlp(net_sizes, True, nn.ReLU, nn.Tanh)
        self.mu_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max
        self.to(device)

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
        self.a_dim = self.env.action_space.n

        self.actor = Actor(self.s_dim, self.hidden_sizes, self.a_dim, -2*self.a_dim, 2*self.a_dim)
        self.actor_target = Actor(self.s_dim, self.hidden_sizes, self.a_dim, -2*self.a_dim, 2*self.a_dim)
        self.critic1 = Critic(self.s_dim+self.a_dim, self.hidden_sizes)
        self.critic1_target = Critic(self.s_dim+self.a_dim, self.hidden_sizes)
        self.critic2 = Critic(self.s_dim+self.a_dim, self.hidden_sizes)
        self.critic2_target = Critic(self.s_dim+self.a_dim, self.hidden_sizes)

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
            s0 = torch.tensor(s0, dtype=torch.float, device=device)
            with torch.no_grad():
                a0, _ = self.actor(s0, deterministic=False, with_logprob=False)
            a0 = torch.argmax(a0)
        return a0.item()

    def learn(self):
        if len(self.buffer.memory) < self.batch_size:
            return

        samples = random.sample(self.buffer.memory, self.batch_size)
        s0, a0, a0_one_hot, r1, s1, d = zip(*samples)
        s0 = torch.stack(s0)  # [batch_size, s_dim]
        a0 = torch.stack(a0).view(self.batch_size, -1)  # [batch_size, 1]
        a0_one_hot = torch.stack(a0_one_hot).view(self.batch_size, -1)  # [batch_size, a_dim]
        s1 = torch.stack(s1)  # [batch_size, s_dim]
        r1 = torch.stack(r1).view(self.batch_size, -1)  # [batch_size, 1]
        d = torch.stack(d).view(self.batch_size, -1)  # [batch_size, 1]

        def critic_learn():
            q1 = self.critic1(s0, a0_one_hot)
            q2 = self.critic2(s0, a0_one_hot)

            with torch.no_grad():
                a1, logp_a1 = self.actor(s1)
                a1_values, a1_indices = torch.max(a1, dim=1, keepdim=True)  # [batch_size, a_dim]
                a1_one_hot = torch.zeros_like(a1).scatter_(1, a1_indices, 1)
                logp_a1_viewed = logp_a1.view(self.batch_size, -1)
                q1_pi_targ = self.critic1_target(s1, a1_one_hot)
                q2_pi_targ = self.critic2_target(s1, a1_one_hot)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                y_true = torch.zeros_like(q1)
                for i in range(self.batch_size):
                    y_true[i] = r1[i] + self.gamma * (1-d[i]) * (q_pi_targ[i] - self.alpha * logp_a1_viewed[i])

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
            logp_pi_viewed = logp_pi.view(self.batch_size, -1)
            logits = F.log_softmax(pi, dim=1)
            differentiable_pi = F.gumbel_softmax(logits, hard=True)
            index = torch.argmax(pi, dim=1, keepdim=True)
            pi_one_hot = torch.zeros_like(pi).scatter_(1, index, 1)  # [batch_size, a_dim]
            q1_pi = self.critic1(s0, differentiable_pi)
            q2_pi = self.critic2(s0, differentiable_pi)
            q_pi = torch.min(q1_pi, q2_pi)

            # Entropy-regularized policy loss
            loss = torch.mean(self.alpha * logp_pi_viewed - q_pi)

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
        'start_steps': 1000,
        'alpha': 0.2,
        'gamma': 0.99,
        'actor_lr': 0.01,
        'critic_lr': 0.01,
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
    https://spinningup.openai.com/en/latest/algorithms/sac.html
    https://arxiv.org/pdf/1812.05905.pdf
    '''
