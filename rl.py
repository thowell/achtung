""" Trains an agent with (stochastic) Policy Gradients on Achtung Die Kurve """
import numpy as np
import pickle 
import os
from achtung_process import AchtungProcess

import matplotlib.pyplot as plt

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 100 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = True
train = True
eval_policy = False

# model initialization
D1 = 80 * 80
D = D1 * 4 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 4.0: running_add = 0 # reset the sum, since this was a game boundary (specific to Achtung reward and frame skip)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

env = AchtungProcess(1)
env.env.render_game = render
env.env.speed = 0
obs = env.reset()

if train:
  observation = env.reset()
  xs,hs,dlogps,drs = [],[],[],[]
  running_reward = None
  reward_sum = 0
  episode_number = 0

  reward_cache = []
  reward_rm_cache = []
  reward_ep_cache = []

  while True:
    if render: env.render()

    x = obs.ravel()

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state
    y = 1 if action == 2 else 0 # a "fake label"
    dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action-1)
    reward_sum += reward

    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

    if done: # an episode finished
      episode_number += 1

      # stack together all inputs, hidden states, action gradients, and rewards for this episode
      epx = np.vstack(xs)
      eph = np.vstack(hs)
      epdlogp = np.vstack(dlogps)
      epr = np.vstack(drs)
      xs,hs,dlogps,drs = [],[],[],[] # reset array memory

      # compute the discounted reward backwards through time
      discounted_epr = discount_rewards(epr)
      # standardize the rewards to be unit normal (helps control the gradient estimator variance)
      discounted_epr -= np.mean(discounted_epr)
      discounted_epr /= np.std(discounted_epr)

      epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
      grad = policy_backward(eph, epdlogp)
      for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

      # perform rmsprop parameter update every batch_size episodes
      if episode_number % batch_size == 0:
        for k,v in model.items():
          g = grad_buffer[k] # gradient
          rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
          model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
          grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

      # boring book-keeping
      running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
      print ('episode (%i) reward: %f. running mean: %f' % (episode_number, reward_sum, running_reward))
      if episode_number % 100 == 0: 
        pickle.dump(model, open('save.p', 'wb'))
        
        reward_ep_cache.append(reward_sum)
        with open("rl_training/reward_ep.txt", "wb") as f:   
          pickle.dump(reward_ep_cache, f)

        reward_rm_cache.append(running_reward)
        with open("rl_training/reward_rm.txt", "wb") as f:   
          pickle.dump(reward_rm_cache, f)

      reward_sum = 0
      observation = env.reset() # reset env
      prev_x = None

if eval_policy:
  env.cache_frames = True
  env.speed = 0
  obs = env.reset()
  n_games = 0
  running_reward = []
  rewards = []

  while n_games < 100:
      cur_x = prepro(obs)
      x = np.concatenate((cur_x, prev_x if prev_x is not None else np.zeros(D1)), axis=None)
      prev_x = cur_x
      aprob, h = policy_forward(x)
      action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

      print("action: ", action)
      
      obs, reward, done, info = env.step(action-1)
      running_reward.append(reward)
      if done:
          obs = env.reset()
          n_games += 1
          rewards.append(sum(running_reward))
          filename = "images/game_{}".format(env.games-1)
          # os.rename(filename, filename + "_{}".format(int(rewards[-1])))
          running_reward = []
          prev_x = None


  print("test complete")
  print("   reward (avg): ", np.mean(rewards))
  print("   reward (std): ", np.std(rewards))
