### 2021-11-30 Shawn: works on DDPG.

- torch.nn.Sequential is compatible with torch.optim.Adam, does not affect convergence arrival.
- Actor Critic's Critic's output_dim===1 and output_activation===torch.nn.Identity, this is critical.
- _Tricks_: for ddpg, continuous learn (batch gradient) is better than stepped learn.

### 2021-12-01 Shawn: works on DDPG.

- For Q-learning, remember output_activation should be Identity. Though you can process the value afterwards, do not introduce other activation function.

### 2021-12-02 Shawn: works on DQN.

- Please differentiate continuous actions and discrete actions. Usually agent act method should be updatd.
- Use .detach() in to ignore gradient descent, improve efficiency.
- Use .gather() to collect values along certain dimension with certain index.

### 2021-12-02 Shawn: works on TD3.

- For TD3, update both critic1 and critic2.

### 2021-12-03 Shawn: works on SAC.

- SAC can be modified for discrete actions.

### 2021-12-04 Shawn: works on SAC.

- For SAC, larger lr (learning rate) and smaller tau (target network update) seem better.
- For SAC, init_wander is less usefull compared to TD3.

### 2021-12-04 Shawn: works on SAC discrete.

- .detach() is not applicable for wrapped functions that return tuple, thus perferring "with torch.no_grad():".
