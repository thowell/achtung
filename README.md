# Achtung Die Kurve!
    
Experiment to train agent to play Achtung Die Kurve! using reinforcement learning.

(1-4 players)

Try the game yourself:
```
python achtung.py
```

Or play with a friend:
```
python achtung.py 2
```

(achtung.py) Environment - Pygame implementation. Initially based on: https://github.com/janowskipio/FarBy, but extensively modified.

(rl.py) Stochastic policy gradient - 2-layer fully-connect neural network policy: based on: http://karpathy.github.io/2016/05/31/rl/

(mu_zero_achtung.py) MuZero - interface to open-source MuZero implementation: https://github.com/werner-duvaud/muzero-general

(train.py) Stochastic policy gradient - simple ResNet policy and stochastic policy gradient implemented in PyTorch, based on: https://github.com/pytorch/examples/tree/master/reinforcement_learning

(a2c/ppo/dqn.ipynb) Actor Critic - Proximal Policy Optimization - Deep Q-Learning - using Stable Baselines 3: https://github.com/pytorch/examples/tree/master/reinforcement_learning

