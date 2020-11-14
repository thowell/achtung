import numpy as np
from achtung import Achtung

env = Achtung()
env.render_game = True
env.speed = 0
env.n_rounds = 1

obs = env.reset()
n_games = 0
running_reward = []
rewards = []

while n_games < 100:
    # Random action
    action = env.action_space.sample() + 1
    print("action: ", action)
    obs, reward, done, info = env.step(action)
    running_reward.append(reward)
    if done:
        obs = env.reset()
        n_games += 1
        rewards.append(sum(running_reward))
        running_reward = []


print("test complete")
print("   reward (avg): ", np.mean(rewards))
print("   reward (std): ", np.std(rewards))
