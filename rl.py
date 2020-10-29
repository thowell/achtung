""" Trains an agent with (stochastic) Policy Gradients on Achtung Die Kurve!. """
import numpy as np
import pickle
from game import *

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = True  # resume from previous checkpoint?
render = True

n_obs = 4 # number of observations used for current "state"
downscale_factor = 2 # image downscale factor

# model initialization
D = int(np.floor(WINDOW_HEIGHT * WINDOW_WIDTH * 3 * n_obs / downscale_factor**2)) # input dimensionality

if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)

# update buffers that add up gradients over a batch
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v)
                 for k, v in model.items()}  # rmsprop memory

def sigmoid(x):
  # sigmoid "squashing" function to interval [0,1]
  return 1.0 / (1.0 + np.exp(-x))

def prepro(I):
#   """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
#   I = I[:195]  # crop
  I = I[::downscale_factor, ::downscale_factor, :]  # downsample by factor
#   I[I == 144] = 0  # erase background (background type 1)
#   I[I == 109] = 0  # erase background (background type 2)
#   I[I != 0] = 1  # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()


def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 1.0:
      # reset the sum, since this was a game boundary (achtung specific!)
      running_add = 0
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h < 0] = 0  # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h  # return probability of taking action 1, and hidden state


def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0  # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1': dW1, 'W2': dW2}

# setup
pygame.init()
env = Achtung(1)
env.speed = 0 # set to zero for training (i.e., no frame delay)
env.render_game = render
observation = env.reset()
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
obs_cache = [prepro(observation) for i in range(n_obs)]

while True:
    
    # preprocess the observation, set input to network to be difference image
    cur_obs = prepro(observation)
    obs_cache.pop()
    obs_cache.insert(0,cur_obs)
    x = np.concatenate(obs_cache, axis=0)

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 0 if np.random.uniform() < aprob else 1  # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    y = 1 if action == 0 else 0  # a "fake label"
    # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    dlogps.append(y - aprob)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    # record reward (has to be done after we call step() to get reward for previous action)
    drs.append(reward)
    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        # if np.std(discounted_epr) != 0.0:
        discounted_epr /= np.std(discounted_epr)

        # modulate the gradient with advantage (PG magic happens right here.)
        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * \
                    rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * \
            0.99 + reward_sum * 0.01
        print('new game: env. episode reward total was %f. (running mean: %f)' %
                (reward_sum, running_reward))
        if episode_number % 100 == 0:
            pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None
