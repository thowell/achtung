import numpy as np
import os
from achtung_process import AchtungProcess

env = AchtungProcess(1)
env.env.render_game = True
env.env.speed = 0

# env.cache_frames = True

obs = env.reset()
n_games = 0
running_reward = []
rewards = []

while n_games < 100:
    # Random action
    action = env.action_space.sample()
    print("action: ", action)
    obs, reward, done, info = env.step(action)
    running_reward.append(reward)
    if done:
        rewards.append(sum(running_reward))
        running_reward = []
        #filename = "images/game_{}".format(env.games-1)
        # os.rename(filename, filename + "_{}".format(int(rewards[-1])))
        obs = env.reset()
        n_games += 1


print("test complete")
print("   reward (avg): ", np.mean(rewards))
print("   reward (std): ", np.std(rewards))
