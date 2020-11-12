from achtung import Achtung

from stable_baselines3 import A2C
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy

env = Achtung()
env.get_action_meanings()
# env.render_game = False
# env.speed = 0

# # Instantiate the agent
# model = A2C('CnnPolicy', env, verbose=1)#, n_steps=int(1e5))
# # Train the agent
# model.learn(total_timesteps=1e5)
# # Save the agent
# model.save("a2c_achtung")
# # del model  # delete trained model to demonstrate loading

# # # Load the trained agent
# # model = A2C.load("a2c_achtung")

# # Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, Achtung(), n_eval_episodes=100)
# print('mean reward: ', mean_reward)
# print('std reward: ', std_reward)

# # # Enjoy trained agent
# # obs = env.reset()
# # games = 0
# # while games < 10:
# #     action, _states = model.predict(obs, deterministic=True)
# #     obs, rewards, done, info = env.step(action)
# #     env.render()

# #     if done:
# #         obs = env.reset()
# #         games += 1