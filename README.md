# Reinforcement Learning

Documentation and implementations for various RL models.

# RL models

![rl-models.png](./images/rl-models.png)

# Classic Envs

## 'CartPole-v1'

```
Observation:
    Type: Box(4)
    Num     Observation               Min                     Max
    0       Cart Position             -4.8                    4.8
    1       Cart Velocity             -Inf                    Inf
    2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
    3       Pole Angular Velocity     -Inf                    Inf
Actions:
    Type: Discrete(2)
    Num   Action
    0     Push cart to the left
    1     Push cart to the right
Episode Termination:
    Pole Angle is more than 12 degrees.
    Cart Position is more than 2.4 (center of the cart reaches the edge of the display).
    Episode length is greater than 200.
    Solved Requirements:
    Considered solved when the average return is greater than or equal to 195.0 over 100 consecutive trials.
```

## 'Pendulum-v1'

_Not explained on Github._

# Env Setup

```
pip install -U autopep8
pip install numpy
pip install gym pygame pyglet
pip3 install torch torchvision torchaudio
pip install matplotlib
```

# TODO

1. Double DQN (https://blog.csdn.net/qq_41871826/article/details/108263919)
2. Dueling DQN
3. Prioritized Experience Replay

# Diaries

### 2021-11-30 Shawn: works on DDPG.

- torch.nn.Sequential is compatible with torch.optim.Adam, does not affect convergence arrival.
- Actor Critic's critic output_dim===1 and output_activation===torch.nn.Identity, this is critical.
- It seems for DDPG, continuous learn (batch gradient) is better than stepped learn.

### 2021-12-01 Shawn: works on DDPG.

- For Q-learning, remember output_activation should be Identity. Though you can process the value afterwards, do not introduce other activation function.
- For all models, 'done' should be considered and learned once.

### 2021-12-02 Shawn: works on DQN.

- Please differentiate continuous actions and discrete actions.
- Use .detach() in to ignore gradient descent, improve efficiency.
- Use .gather() to collect values from input along certain dimension with certain index. Return has the same shape with index.

### 2021-12-02 Shawn: works on TD3.

- It seems for TD3, update both critic1 and critic2 in .actor_learn() is good, use torch.min to align with .critic_learn().

### 2021-12-03 Shawn: works on SAC.

- SAC can be modified for discrete actions.

### 2021-12-04 Shawn: works on SAC.

- For SAC, larger lr (learning rate) and smaller tau (target network update) seem better.

### 2021-12-05 Shawn: works on DDPG discrete.

- .detach() is not applicable for wrapped functions that return tuple, thus perferring "with torch.no_grad():".
- It seems for DDPG/TD3 discrete, smaller gamma is good; for SAC discrete, larger gamma is good.

### 2021-12-06 Shawn: works on DDPG discrete.

- Use .scatter() to place src values by index to Tensor.

### 2021-12-07 Shawn: works on DDPG discrete.

- Use .gumbel_softmax(logits, hard=True) as candidate for .argmax() + .scatter(). One-hot encoding is required, but .scatter() is undifferentiable.
- For DDPG/TD3 discrete, add one-hot data to .actor_learn() to avoid local optima if necessary.
- For SAC discrete, do not add one-hot data to .actor_learn().

### 2021-12-09 Shawn: works on all models.

- It seems for all models, either long start_steps or eps-decay-random-action is good for exploration.
- DDPG, TD3, SAC exist local optima issue, even after long training. It seems long start_steps helps resolve this issue. Reducing network also helps resolve this issue.

### 2021-12-12 Shawn: works on all models.

- python render functions are power-consuming, disable if necessary.
- For TD3, .critic_learn() by random critic, .actor_learn() by best critic, to achieve better control target, and to avoid local optima.

### 2021-12-15 Shawn: works on A2C.

- It seems for A2C, larger gamma is good, and larger lr is good for faster convergence.
- policy-based model (A2C) and value-based model (DQN) can cross evaluate.

### 2021-12-16 Shawn: works on A2C.

- It seems for A2C, positive reward is better. And larger lr at beginning and smaller lr afterwards is best.
- For A2C, larger entropy leads to exploration/oscillation.

### 2022-02-19 Shawn: works on all models.

- store all tensors, including a0 one hot for discrete models, for faster training.
