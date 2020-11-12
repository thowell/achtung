import argparse
from game import *
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
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-interval', type=int, default=10000, metavar='N',
                    help='interval between training status logs (default: 1000)')
args = parser.parse_args()


pygame.init()
env = Achtung(1)
env.speed = 0 # set to zero for training (i.e., no frame delay)
env.render_game = False
downscale_factor = 2

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p_drop=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=5, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.drop = nn.Dropout(p=p_drop)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, p_drop):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.p_drop = p_drop
        self.input_channels = 1
        output_num = 3

        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=9, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=2)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(1152*block.expansion, output_num)
        self.softmax = nn.Softmax(dim=-1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, p_drop=self.p_drop))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = torch.flatten(out, 1)
        #print(out.shape)
        out = self.linear(out)
        out = self.softmax(out)
        return out



class Policy():
    def __init__(self):
        super(Policy, self).__init__()
        # self.affine1 = nn.Linear(4, 128)
        # self.dropout = nn.Dropout(p=0.6)
        # self.affine2 = nn.Linear(128, 2)

        self.gamma = args.gamma
        # Episode policy and reward history 
        self.policy_history = Variable(torch.Tensor()).to(device) 
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.net = ResNet(BasicBlock, [1], 0.1)

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

policy = Policy()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    # self.net = nn.DataParallel(self.net)
policy.net.to(device)

optimizer = optim.Adam(policy.net.parameters(), lr=8e-5)
# optimizer = optim.SGD(policy.net.parameters(), lr=1e-2, momentum=0.9)
eps = np.finfo(np.float32).eps.item()

def prepro(I):
    I = I[::downscale_factor, ::downscale_factor, 0]  # downsample by factor
    I = I / (2*255)
    I[:,0] = 1
    I[:,-1] = 1
    I[0,:] = 1
    I[-1,:] = 1
    return torch.from_numpy(I.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def select_action(state):
    state = prepro(state) 
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
    running_reward = 10
    episode_length = []
    for i_episode in count(1):
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
        update_policy()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(i_episode, t, running_reward))
        if running_reward > 250:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, t))
            break

        if (i_episode % args.save_interval == 0 or i_episode % args.save_interval-1 == 0 or i_episode % args.save_interval-2 == 0):
            np.save("images/ep" + str(i_episode) + "_" + str(t), state)

        if (i_episode % args.save_interval == 0 ):
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