### 2021-11-30 Shawn: works on ddpg.

- torch.nn.Sequential is compatible with torch.optim.Adam, does not affect convergence arrival.
- Actor Critic's Critic's output_dim===1 and output_activation===torch.nn.Identity, this is critical.
- _Trick_: for ddpg, continuous learn (batch gradient) is better than stepped learn.

### 2021-12-01 Shawn: works on ddpg.
