import argparse
from achtung_process import AchtungProcess
import numpy as np
from itertools import count
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1000)')
args = parser.parse_args()


env = AchtungProcess(1)
env.env.speed = 0 # set to zero for training (i.e., no frame delay)
env.env.render_game = True
obs = env.reset()
batch_size = 100

class CNN(nn.Module):
    def __init__(self, h, w, c, outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(c, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        self.softmax = nn.Softmax(dim=-1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        x = self.softmax(x)
        return x

class Policy():
    def __init__(self, h, w, c, na):
        super(Policy, self).__init__()

        self.gamma = args.gamma
        # Episode policy and reward history 
        self.policy_history = Variable(torch.Tensor()).to(device) 
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.net = CNN(h, w, c, na)

    def dump(self, model_file):
        if torch.cuda.device_count() > 1:
            torch.save(self.net.module, model_file)
        else:
            torch.save(self.net, model_file)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Training on GPU \n")
else:
    device = torch.device("cpu")
    print("Training on CPU")

# y = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
# print(y.shape)
# model = CNN(80,80,4,2)
# model.forward(y)
policy = Policy(80,80,4,3)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
policy.net.to(device)

optimizer = optim.Adam(policy.net.parameters(), lr=1.0e-5)
# optimizer = optim.SGD(policy.net.parameters(), lr=1e-2, momentum=0.9)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state.astype(np.float32)).unsqueeze(0) 
    state = state.to(device)
    probs = policy.net(state)
    m = Categorical(probs)
    action = m.sample()
    policy.policy_history = torch.cat([policy.policy_history, m.log_prob(action)])
    return action.item()

def update_policy():
    R = 0
    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)
        
    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards).to(device)).mul(-1), -1))
    
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor()).to(device) 
    policy.reward_episode= []


def main():
    running_reward = 10.0
    episode_length = []
    i_episode = 0
    for i in count(1):
        for k in range(batch_size):
            print("episode:", len(episode_length))
            state, ep_reward = env.reset(), 0
            for t in range(1, 1000):  # Don't infinite loop while learning
                action = select_action(state)
                state, reward, done, _ = env.step(action)
                if args.render:
                    env.render()
                policy.reward_episode.append(reward)
                ep_reward += reward
                if done:
                    break
            # Used to determine when the environment is solved.
            episode_length.append(t)
            running_reward = (running_reward * 0.95) + (ep_reward * 0.05)
            print('   reward:', ep_reward)
            print('   running reward:', running_reward)
            i_episode += 1
        print("update policy")
        update_policy()

        if running_reward > 250:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, t))
            break

        # if (i_episode % 1000 == 0 or i_episode % 1000-1 == 0 or i_episode % 1000-2 == 0):
        #     np.save("images/ep" + str(i_episode) + "_" + str(t), state)

        if (i_episode % 1000 == 0):
            print('\n Saving checkpoint ' + "net_" + str(i_episode) + ".ptmodel\n")
            policy.dump("models/ep" + str(i_episode) + ".ptmodel")
            with open("models/loss.txt", "wb") as f:   
                pickle.dump(policy.loss_history, f)
            with open("models/reward.txt", "wb") as f:   
                pickle.dump(policy.reward_history, f)
            with open("models/length.txt", "wb") as f:   
                pickle.dump(episode_length, f) 

if __name__ == '__main__':
    main()
