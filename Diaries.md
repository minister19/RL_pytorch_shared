### 2021-11-30 Shawn: works on ddpg.

- torch.nn.Sequential is compatible with torch.optim.Adam, does not affect convergence arrival.
- Actor Critic's Critic's output_dim===1 and output_activation===torch.nn.Identity, this is critical.
- _Tricks_: for ddpg, continuous learn (batch gradient) is better than stepped learn.

### 2021-12-01 Shawn: works on ddpg.

- For Q-learning, remember output_activation should be Identity. Though you can process the value afterwards, do not introduce other activation function.

### 2021-12-02 Shawn: works on dqn.

- Please differentiate continuous actions and discrete actions. Usually agent act method should be updatd.
- Use .detach() in to ignore gradient descent, improve efficiency.
- Use .gather() to collect values along certain dimension with certain index.

### 2021-12-02 Shawn: works on td3.

- For TD3, update both critic1 and critic2.
