from achtung import Achtung

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy

env = Achtung()
env.render_game = True
env.speed = 0

# Instantiate the agent
model = PPO('CnnPolicy', env, verbose=1)
# Train the agent
print("HI1")
model.learn(log_interval=10,total_timesteps=int(2e5))
# Save the agent
print("HI2")
model.save("ppo_achtung")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = PPO.load("ppo_achtung")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print('mean reward: ', mean_reward)
print('std reward: ', std_reward)

# Enjoy trained agent
obs = env.reset()
while games < 10:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    env.render()

    if done:
        games += 1