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

from Agent.Actor import Actor
from Agent.Critic import Critic
from util.Grid import MARGIN, Map

BATCH_SIZE = 256
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


class ActorCriticAgent():
    def __init__(self):
        # if gpu is to be used
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.actor = Actor().to(device).double()
        self.critic = Critic().to(device).double()

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.memory = ReplayMemory(100000)

        self.steps_done = 0

        self.scale = 20.0
        self.map_width = int(8*self.scale)
        self.map_height = int(5*self.scale)
        icra_map = Map(self.map_width, self.map_height)
        grid = icra_map.getGrid()
        self.obs_map = torch.from_numpy(1-grid)[MARGIN:-MARGIN, MARGIN:-MARGIN]
        self.whole_map = torch.from_numpy(np.load("resources/ob.npy"))[
            MARGIN:-MARGIN, MARGIN:-MARGIN]
        #self.whole_map = torch.from_numpy(1-grid).to(device)

        # self.view_width, self.view_height = 0.8, 0.5 # m(half)
        #self.grid_width, self.grid_height = int(self.map_width*(self.view_width*2/8)), int(self.map_height*(self.view_height*2/5))

        x, y = np.mgrid[0:5.0:5.0/100, 0:8.0:8.0/160]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        self.pos = pos
        #rv = multivariate_normal([0.1, 0.1], [[0.01, 0.0], [0.0, 0.01]])
        #gaussian = torch.from_numpy(rv.pdf(pos)).to(device).double()
        #self.gaussian = gaussian.unsqueeze(0).unsqueeze(0).repeat(BATCH_SIZE,1,1,1)
        #self.gaussian = gaussian
        #self.whole_rand = torch.rand(self.grid_height, self.grid_width).double().to(device)
        #self.value_map = torch.zeros(1, 1, self.grid_height, self.grid_width).double().to(device)
        #plt.imshow(self.gaussian, vmin=0, cmap="GnBu")
        # plt.show()

    def perprocess_state(self, state):
        p = state[:2]
        e_p = state[-2:]
        '''
        left, bottom = p[0]-self.view_width, p[1]-self.view_height
        left, bottom = int(left*self.scale), int(bottom*self.scale)
        left += MARGIN
        bottom += MARGIN
        right = left + self.grid_width
        top = bottom + self.grid_height
        self.window = (left, right, bottom, top)
        window_map = (self.whole_map[bottom:top, left:right])
        enemy_map = (torch.zeros((self.grid_height), (self.grid_width)).to(device).double())
        '''
        window_map = self.whole_map
        pos = self.pos
        rv = multivariate_normal(p[::-1], [[0.01, 0.0], [0.0, 0.01]])
        pos_map = torch.from_numpy(rv.pdf(pos)).double()
        rv = multivariate_normal(e_p[::-1], [[0.01, 0.0], [0.0, 0.01]])
        enemy_map = torch.from_numpy(rv.pdf(pos)).double()
        '''
        if e_p[0] > 0:
            ENEMY_SIZE = 2
            delta_pos = (np.array(e_p) - np.array(p))
            delta_pos = np.clip(delta_pos, [-0.8, -0.5], [0.8, 0.5])
            #x, y = np.mgrid[-0.5:0.5:1/self.scale,
                            #-0.8:0.8:1/self.scale]
            #pos = np.empty(x.shape + (2,))
            #pos[:, :, 0] = x; pos[:, :, 1] = y
            pos = self.pos
            rv = multivariate_normal(delta_pos[::-1], [[0.01, 0.0], [0.0, 0.01]])
            enemy_map = torch.from_numpy(rv.pdf(pos)).to(device).double() / 100.0
        '''
        if False:
            plt.cla()
            #plt.xlim(0, self.grid_width-1)
            #plt.ylim(0, self.grid_height-1)
            plt.imshow(pos_map, vmin=0, cmap="GnBu")
            plt.pause(0.0001)
        #window_map = window_map.unsqueeze(0).unsqueeze(1)
        #pos_map = pos_map.unsqueeze(0).unsqueeze(1)
        #enemy_map = enemy_map.unsqueeze(0).unsqueeze(1)
        state_map = torch.stack(
            [window_map, pos_map, enemy_map], dim=0)  # 3, h, w
        return state_map.unsqueeze(0)

    def select_action(self, state_map, mode):
        device = self.device

        with torch.no_grad():
            self.actor.eval()
            state_map = state_map.to(device)
            x, y = self.actor(state_map)
            v = self.critic(state_map)
            #print(v)
        # mu: [0, 1]
        #mu, std = mu.cpu().numpy()[0], std.cpu().numpy()[0]
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
            #x = mu[0] * 8.0
            #y = mu[1] * 5.0
        elif mode == "sample":
            x = np.random.choice(range(32),p=x)
            y = np.random.choice(range(20),p=y)
            #sample = mu + std * np.random.randn(2)
            #x = sample[0] * 8.0
            #y = sample[1] * 5.0

        x = x / 32.0 * 8.0
        y = y / 20.0 * 5.0
        return [x, y]

    def push(self, state, next_state, goal, reward):
        device = self.device
        # goal: [0, 8], [0, 5]
        normalized_goal = [goal[0] / 8.0, goal[1] / 5.0]
        #goal = torch.tensor(goal).to(device).double().unsqueeze(0)
        #next_state = torch.tensor(next_state).to(device).double().unsqueeze(0)
        #state = torch.tensor(state).to(device).double().unsqueeze(0)
        #reward = torch.tensor([reward]).to(device).double().unsqueeze(0)
        self.memory.push(state, normalized_goal, next_state, reward)

    def make_state_map(self, state):
        state_map = []
        window_map = self.whole_map
        pos = self.pos
        for p, e_p in state:
            rv = multivariate_normal(p[::-1], [[0.01, 0.0], [0.0, 0.01]])
            pos_map = torch.from_numpy(rv.pdf(pos)).double()
            rv = multivariate_normal(e_p[::-1], [[0.01, 0.0], [0.0, 0.01]])
            enemy_map = torch.from_numpy(rv.pdf(pos)).double()

            state_map.append(torch.stack(
                [window_map, pos_map, enemy_map], dim=0))  # 3, h, w
        state_map = torch.stack(state_map, dim=0)  # batch, 3, h, w
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

        self.actor.train()
        ### Critic ###
        value_eval = self.critic(state_batch)
        #next_value_eval = self.critic(next_state_batch)
        #td_error = (reward_batch + 2) / 1000.0 + GAMMA*next_value_eval - value_eval
        #loss = torch.sum(td_error ** 2)
        td_error = reward_batch / 4000.0 - value_eval
        loss = F.smooth_l1_loss(value_eval, reward_batch / 4000.0)
        self.optimizer_critic.zero_grad()
        loss.backward()
        for param in self.critic.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer_critic.step()

        ### Actor ###
        state_batch = Variable(state_batch, requires_grad=True)
        x, y = self.actor(state_batch)  # batch, 1, 10, 16
        #plt.cla()
        #plt.imshow(x.detach().numpy())
        #plt.pause(0.00001)
        prob = x.gather(1, (action_batch[:,0:1]*32).long()) * y.gather(1, (action_batch[:,1:2]*20).long())
        loss = -prob.sum()
        #prob = normal(action_batch, mu, std)
        #action_batch = action_batch[:,1:]*state_action_prob.size(3)+action_batch[:,:1]
        #state_action_prob = state_action_prob.reshape([BATCH_SIZE, -1])
        #state_action_prob = state_action_prob.gather(1, action_batch)
        #log_prob = torch.log(prob)
        #exp_v = torch.mean(log_prob * td_error.detach())
        #loss = -exp_v
        self.optimizer_actor.zero_grad()
        loss.backward()
        for param in self.actor.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer_actor.step()

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
            self.actor.eval()
            ### Critic ###
            value_eval = self.critic(state_batch)
            #next_value_eval = self.critic(next_state_batch)
            #td_error = reward_batch / 100.0 + GAMMA*next_value_eval - value_eval
            td_error = reward_batch / 4000.0 - value_eval
            loss_c = F.smooth_l1_loss(value_eval, reward_batch / 4000.0)

            ### Actor ###
            x, y = self.actor(state_batch)  # batch, 1, 10, 16
            prob = x.gather(1, (action_batch[:,0:1]*32).long()) * y.gather(1, (action_batch[:,1:2]*20).long())
            #prob = normal(action_batch, mu, std)
            log_prob = torch.log(prob)
            exp_v = torch.mean(log_prob * td_error.detach())
            loss_a = -exp_v

        return loss_c.item(), loss_a.item()

    def save_model(self):
        torch.save(self.actor.state_dict(), "Actor.model")
        torch.save(self.critic.state_dict(), "Critic.model")

    def save_memory(self, file_name):
        torch.save(self.memory, file_name)

    def load_model(self):
        self.actor.load_state_dict(torch.load(
            "Actor.model", map_location=self.device))
        self.critic.load_state_dict(torch.load(
            "Critic.model", map_location=self.device))

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
