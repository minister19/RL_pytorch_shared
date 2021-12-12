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
- It seems for DDPG/TD3 discrete models, smaller gamma is good; for SAC discrete, larger gamma is good.

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
