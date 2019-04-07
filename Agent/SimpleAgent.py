import math
import random
import time
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import multivariate_normal
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from util.Grid import MARGIN, Map

BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
        )
        self.head_x = nn.Sequential(
            nn.Linear(64, 32),
            nn.Softmax(dim=1)
        )
        self.head_y = nn.Sequential(
            nn.Linear(64, 20),
            nn.Softmax(dim=1)
        )
    def forward(self, s):
        feature = self.fc(s)
        #batch, _ = feature.shape
        #feature = feature.reshape([batch, -1])
        x, y = self.head_x(feature), self.head_y(feature)
        return x, y

def normal(x, mu, std):
    a = (-1*(x-mu).pow(2)/(2*std)).exp()
    b = 1/(2*std*np.pi).sqrt()
    return a*b


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.epoch_memory = []
        self.main_memory = []
        self.position = 0
        #self.epoch_begin = 0

    def push(self, *args):
        """Saves a transition."""
        self.epoch_memory.append(Transition(*args))
        #if len(self.main_memory) < self.capacity:
            #self.main_memory.append(None)
        #self.main_memory[self.position] = Transition(*args)
        #self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, is_test=False):
        if is_test:
            return self.main_memory[0:batch_size]
        else:
            return random.sample(self.main_memory, batch_size)

    def finish_epoch(self):
        R = 0
        if len(self.main_memory) < self.capacity - len(self.epoch_memory):
            for t in self.epoch_memory[::-1]:
                state, action, next_state, reward = t
                reward = reward[0]
                R = reward + GAMMA*R
                self.main_memory.append(Transition(state, action, next_state, [R]))
        else:
            for t in self.epoch_memory[::-1]:
                state, action, next_state, reward = t
                reward = reward[0]
                R = reward + GAMMA*R
                self.main_memory[self.position] = Transition(state, action, next_state, [R])
                self.position = (self.position + 1) % self.capacity
        del self.epoch_memory
        self.epoch_memory = []

    def __len__(self):
        return len(self.main_memory)

    def __getitem__(self, index):
        return self.main_memory[index]


class SimpleAgent():
    def __init__(self):
        # if gpu is to be used
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = Simple().to(device).double()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = ReplayMemory(100000)

    def perprocess_state(self, state):
        p = state[:2]
        e_p = state[-2:]
        if False:
            plt.cla()
            #plt.xlim(0, self.grid_width-1)
            #plt.ylim(0, self.grid_height-1)
            plt.imshow(pos_map, vmin=0, cmap="GnBu")
            plt.pause(0.0001)

        state_map = torch.tensor([p+e_p]).double()
        return state_map

    def select_action(self, state_map, mode):
        device = self.device

        with torch.no_grad():
            state_map = state_map.to(device)
            x, y = self.model(state_map)

        x = x.cpu().numpy()
        y = y.cpu().numpy()
        if True:
            plt.cla()
            plt.imshow(x)
            plt.pause(0.00001)

        x, y = x[0], y[0]

        if mode == "max_probability":
            x = np.argmax(x)
            y = np.argmax(y)
        elif mode == "sample":
            x = np.random.choice(range(32),p=x)
            y = np.random.choice(range(20),p=y)

        x = x / 32.0 * 8.0
        y = y / 20.0 * 5.0
        return [x, y]

    def push(self, state, next_state, goal, reward):
        device = self.device
        # goal: [0, 8], [0, 5]
        normalized_goal = [goal[0] / 8.0, goal[1] / 5.0]
        self.memory.push(state, normalized_goal, next_state, reward)

    def make_state_map(self, state):
        state_map = torch.tensor(state).double()
        return state_map

    def sample_memory(self, is_test=False):
        device = self.device
        transitions = self.memory.sample(BATCH_SIZE, is_test)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        #state_batch = torch.cat(batch.state).to(device)
        #next_state_batch = torch.cat(batch.next_state).to(device)
        state_batch = self.make_state_map(batch.state).double()
        next_state_batch = self.make_state_map(batch.next_state).double()
        action_batch = torch.tensor(batch.action).double()
        reward_batch = torch.tensor(batch.reward).double()

        return state_batch, action_batch, reward_batch, next_state_batch

    def optimize_once(self, data):
        state_batch, action_batch, reward_batch, next_state_batch = data
        device = self.device
        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        next_state_batch = next_state_batch.to(device)

        state_batch = Variable(state_batch, requires_grad=True)
        x, y = self.model(state_batch)  # batch, 1, 10, 16
        print(x.shape, y.shape, action_batch.shape)
        prob = x.gather(1, (action_batch[:,0:1]*32).long()) * y.gather(1, (action_batch[:,1:2]*20).long())
        log_prob = torch.log(prob)
        exp_v = torch.mean(log_prob * reward_batch)
        loss = -exp_v
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def optimize_online(self):
        if len(self.memory) < BATCH_SIZE:
            return
        data = self.sample_memory()
        loss = self.optimize_once(data)
        return loss

    def test_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        state_batch, action_batch, reward_batch, next_state_batch = self.sample_memory(True)
        device = self.device
        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        next_state_batch = next_state_batch.to(device)

        with torch.no_grad():
            x, y = self.model(state_batch)  # batch, 1, 10, 16
            prob = x.gather(1, (action_batch[:,0:1]*32).long()) * y.gather(1, (action_batch[:,1:2]*20).long())
            log_prob = torch.log(prob)
            exp_v = torch.mean(log_prob * reward_batch)
            loss = -exp_v

        return loss.item()

    def save_model(self):
        torch.save(self.model.state_dict(), "Simple.model")

    def save_memory(self, file_name):
        torch.save(self.memory, file_name)

    def load_model(self):
        self.model.load_state_dict(torch.load(
            "Simple.model", map_location=self.device))

    def load_memory(self, file_name):
        self.memory = torch.load(file_name)

    def optimize_offline(self, num_epoch):
        def batch_state_map(transitions):
            batch = Transition(*zip(*transitions))
            state_batch = self.make_state_map(batch.state).double()
            next_state_batch = self.make_state_map(batch.next_state).double()
            action_batch = torch.tensor(batch.action).double()
            reward_batch = torch.tensor(batch.reward).double()
            return state_batch, action_batch, reward_batch, next_state_batch

        dataloader = DataLoader(self.memory.main_memory, batch_size=64,
                                shuffle=True, collate_fn=batch_state_map, num_workers=10, pin_memory=True)
        device = self.device
        for epoch in range(num_epoch):
            print("Train epoch: [{}/{}]".format(epoch, num_epoch))
            for data in tqdm(dataloader):
                loss = self.optimize_once(data)
            loss = self.test_model()
            print("Test loss: {}".format(loss))
        return loss
