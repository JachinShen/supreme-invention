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

from Agent.Actor import ActorCritic
from util.Grid import MARGIN, Map

BATCH_SIZE = 2048
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


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
        if len(self.main_memory) < self.capacity:
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


class ActorCriticAgent():
    def __init__(self):
        # if gpu is to be used
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.policy_net = ActorCritic().to(device).double()
        self.target_net = ActorCritic().to(device).double()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.lr = 1e-5
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(100000)

    def perprocess_state(self, state):
        return torch.tensor([state]).double() # 1x2xseq

    def select_action(self, state_map, mode):
        device = self.device

        with torch.no_grad():
            self.target_net.eval()
            state_map = state_map.to(device)
            a, v = self.target_net(state_map)
            #print(v)
        a = a.cpu().numpy()[0]

        if mode == "max_probability":
            a = np.argmax(a)
        elif mode == "sample":
            a += 0.01
            a /= a.sum()
            a = np.random.choice(range(6),p=a)
        return a

    def push(self, state, next_state, action, reward):
        self.memory.push(state, action, next_state, reward)

    def make_state_map(self, state):
        return torch.tensor(state).double() # batchx2xseq

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

        self.policy_net.train()
        state_batch = Variable(state_batch, requires_grad=True)
        a, value_eval = self.policy_net(state_batch)  # batch, 1, 10, 16
        ### Critic ###
        td_error = reward_batch - value_eval
        ### Actor ###
        #prob = x.gather(1, (action_batch[:,0:1]*32).long()) * y.gather(1, (action_batch[:,1:2]*20).long())
        prob = a.gather(1, action_batch.long())
        log_prob = torch.log(prob + 1e-6)
        exp_v = torch.mean(log_prob * td_error.detach())
        loss = -exp_v + F.smooth_l1_loss(value_eval, reward_batch)
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.model.parameters():
            #if param.grad is not None:
                #param.grad.data.clamp_(-1, 1)
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
            self.target_net.eval()
            a, value_eval = self.target_net(state_batch)  # batch, 1, 10, 16
            ### Critic ###
            td_error = reward_batch - value_eval
            ### Actor ###
            #prob = x.gather(1, (action_batch[:,0:1]*32).long()) * y.gather(1, (action_batch[:,1:2]*20).long())
            prob = a.gather(1, action_batch.long())
            log_prob = torch.log(prob)
            exp_v = torch.mean(log_prob * td_error.detach())
            loss = -exp_v + F.smooth_l1_loss(value_eval, reward_batch)

        return loss.item()

    def save_model(self):
        torch.save(self.policy_net.state_dict(), "Actor.model")

    def save_memory(self, file_name):
        torch.save(self.memory, file_name)

    def load_model(self):
        self.policy_net.load_state_dict(torch.load(
            "Actor.model", map_location=self.device))

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

        dataloader = DataLoader(self.memory.main_memory, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=batch_state_map, num_workers=10, pin_memory=True)
        device = self.device
        for epoch in range(num_epoch):
            #print("Train epoch: [{}/{}]".format(epoch, num_epoch))
            for data in (dataloader):
                loss = self.optimize_once(data)
            #loss = self.test_model()
            #print("Test loss: {}".format(loss))
        return loss
    
    def decay_LR(self, decay):
        self.lr *= decay
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())