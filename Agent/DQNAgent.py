'''
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
import math
import random
from collections import namedtuple
from itertools import count

import Box2D
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import multivariate_normal
from torch.autograd import Variable

import cv2
from Agent.DQN import DQN
from SupportAlgorithm.DynamicWindow import DynamicWindow
from SupportAlgorithm.GlobalLocalPlanner import GlobalLocalPlanner
from SupportAlgorithm.NaiveMove import NaiveMove
from util.Grid import Map

BATCH_SIZE = 128
#GAMMA = 0.999
GAMMA = 0.5
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent():
    def __init__(self):
        # if gpu is to be used
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.policy_net = DQN().to(device).double()
        self.target_net = DQN().to(device).double()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-2)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        #self.target = (-10, -10)
        #self.move = GlobalLocalPlanner()
        #self.move = NaiveMove()

        self.scale = 20.0
        self.map_width = int(8*self.scale)
        self.map_height = int(5*self.scale)
        icra_map = Map(self.map_width, self.map_height)
        grid = icra_map.getGrid()
        self.whole_map = torch.from_numpy(1-grid).to(device)
        self.obs_map = torch.from_numpy(grid*-1e10).to(device)

        self.view_width, self.view_height = 0.8, 0.5 # m
        self.grid_width, self.grid_height = int(self.map_width*(self.view_width*2/8)), int(self.map_height*(self.view_height*2/5))

        x, y = np.mgrid[-0.5:0.5:1/self.scale, -0.8:0.8:1/self.scale]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        rv = multivariate_normal([0.0, -0.0], [[0.1, 0.0], [0.0, 0.1]])
        gaussian = torch.from_numpy(rv.pdf(pos)).to(device).double()
        self.gaussian = gaussian.unsqueeze(0).unsqueeze(0).repeat(BATCH_SIZE,1,1,1)
        #plt.imshow(rv.pdf(pos))
        #plt.show()

    def perprocess_state(self, state):
        device = self.device
        p = state[:2]
        e_p = state[-2:]
        left, bottom = p[0]-self.view_width, p[1]-self.view_height
        left, bottom = int(left*self.scale), int(bottom*self.scale)
        right = left + self.grid_width
        top = bottom + self.grid_height
        if left < 0:
            right = self.grid_width
            left = 0
        if right > self.map_width:
            left = self.map_width - self.grid_width
            right = self.map_width
        if bottom < 0:
            top = self.grid_height
            bottom = 0
        if top > self.map_height:
            bottom = self.map_height - self.grid_height
            top = self.map_height
        self.window = (left, right, bottom, top)
        window_map = (self.whole_map[bottom:top, left:right])
        enemy_map = (torch.zeros((self.grid_height), (self.grid_width)).to(device).double())
        if e_p[0] > 0:
            delta_pos = (np.array(e_p) - np.array(p))*self.scale
            delta_pos = np.clip(delta_pos, [2, 2], [self.grid_width-3, self.grid_height-3]).astype("int")
            enemy_map[delta_pos[1]-2:delta_pos[1]+2, delta_pos[0]-2:delta_pos[0]+2] = 1
        window_map = window_map.unsqueeze(0)
        enemy_map = enemy_map.unsqueeze(0)
        state = torch.cat([window_map, enemy_map], dim=0).unsqueeze(0) # 1, 2, 5, 8
        return state

    def select_action(self, state, state_obs, is_test=False):
        device = self.device
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        pos = (state[0], state[1])
        vel = (state[2], state[3])
        angle = state[4]
        angular = state[5]
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        left, right, bottom, top = self.window
        self.state_obs = state_obs
        '''
        state = torch.tensor(state).to(device).unsqueeze(0).double()
        width, height = 1.6/2, 1.0/2
        left, right, bottom, top = pos[0]-width, pos[0]+width, pos[1]-height, pos[1]+height
        left, right, bottom, top = int(left*5), int(right*5), int(bottom*5), int(top*5)
        #left, right, bottom, top = max(0,left), min(40,right), max(0,bottom), min(25,top)
        if left < 0:
            right -= left
            left = 0
        if right > 40:
            left -= (right-40)
            right = 40
        if bottom < 0:
            top -= bottom
            bottom = 0
        if top > 25:
            bottom -= (top-25)
            top = 25
        '''
        if sample > eps_threshold:
            with torch.no_grad():
                self.policy_net.eval()
                value_map = self.policy_net(state_obs)[0][0]
                value_map += self.obs_map[bottom:top, left:right]

                if is_test:
                    plt.cla()
                    plt.xlim(0,self.grid_width-1)
                    plt.ylim(0,self.grid_height-1)
                    #plt.imshow(self.grid_final[bottom:top, left:right])
                    plt.imshow(np.log(value_map.numpy()+1e9), cmap="coolwarm")
                    #plt.imshow(value_map.numpy(), cmap="coolwarm")
                    plt.pause(0.00001)
                col_max, col_max_indice = value_map.max(dim=0)
                max_col_max, max_col_max_indice = col_max.max(dim=0)
                x = max_col_max_indice.item()
                y = col_max_indice[x].item()
                self.local_goal = [x, y]
                x += left
                y += bottom
                x = x/self.map_width*8.0
                y = y/self.map_height*5.0

        else:
            value_map = torch.rand(top-bottom, right-left).double().to(device)
            value_map -= self.gaussian[0][0]
            #plt.imshow(value_map.numpy())
            #plt.show()
            value_map += self.obs_map[bottom:top, left:right]
            col_max, col_max_indice = value_map.max(dim=0)
            max_col_max, max_col_max_indice = col_max.max(dim=0)
            x = max_col_max_indice.item()
            y = col_max_indice[x].item()
            self.local_goal = [x, y]
            x += left
            y += bottom
            x = x/self.map_width*8.0
            y = y/self.map_height*5.0
            #col_max, col_max_indice = value_map.max(0)
            #max_col_max, max_col_max_indice = col_max.max(0)
            #x = max_col_max_indice.item()
            #y = col_max_indice[x].item()
            ##x = x/40.0*8.0
            #y = y/25.0*5.0
        
   #x, y = random.random()*8.0, random.random()*5.0

        return [x, y]
        '''
        self.target = (x, y)
        try:
            self.move.setGoal(pos, self.target)
        except:
            return action
        self.steps_done += 1

        action = self.move.moveTo(pos, vel, angle, angular, action)
        return action
        '''

    def push(self, next_state, reward):
        device = self.device
        goal = torch.Tensor(self.local_goal).to(device).long().unsqueeze(0)
        #target = torch.tensor(self.target, device=device).double()
        #next_state = torch.tensor(next_state).to(device).double()
        next_state = (next_state).to(device).double()
        reward = torch.tensor([reward], device=device).double().unsqueeze(0)
        self.memory.push(self.state_obs, goal, next_state, reward)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        device = self.device
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_batch = Variable(state_batch, requires_grad=True)
        self.policy_net.train()
        state_action_values = self.policy_net(state_batch) # batch, 1, 10, 16
        #regular_term = 1e-1*state_action_values[:,:,2:8, 7:11].mean()
        #regular_term = 1e-8 * (state_action_values*self.gaussian).sum()
        action_batch = action_batch[:,1:]*state_action_values.size(3)+action_batch[:,:1]
        state_action_values = state_action_values.reshape([BATCH_SIZE, -1])
        state_action_values = state_action_values.gather(1, action_batch)

        #regular_term = 1e-6 * (state_action_values**2).sum()
        #state_action_values = state_action_values.max(dim=1)[0]

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device).double()
        value = self.target_net(non_final_next_states)
        value = value.reshape([BATCH_SIZE, -1]).max(dim=1)[0].detach()
        next_state_values[non_final_mask] = value
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * GAMMA) + reward_batch * 1e-2

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values)

        #loss += regular_term

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self):
        torch.save(self.policy_net.state_dict(), "ICRA.model")

    def load(self):
        self.policy_net.load_state_dict(torch.load(
            "ICRA.model", map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
