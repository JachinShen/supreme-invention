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

import cv2
from Agent.DQN import DQN
from SupportAlgorithm.GlobalLocalPlanner import GlobalLocalPlanner
from util.Grid import Map

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

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

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.target = (-10, -10)
        self.move = GlobalLocalPlanner()
        icra_map = Map(40, 25)
        grid = icra_map.getGrid()
        #grid = cv2.resize(grid, (17, 17), interpolation=cv2.INTER_AREA)
        grid = 1 - grid
        self.grid = torch.from_numpy(grid).to(device)

    def select_action(self, state, is_test=False):
        device = self.device
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if state[-1] > 0 and state[-3] > 0:
            action[4] = +1.0
        else:
            action[4] = 0.0

        pos = (state[0], state[1])
        vel = (state[2], state[3])
        angle = state[4]
        angular = state[5]
        state = torch.tensor(state).to(device).unsqueeze(0).double()
        self.state = state
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        if is_test or sample > eps_threshold:
            with torch.no_grad():
                value_map = self.policy_net(state)[0][0]
                value_map *= self.grid
                # plt.imshow(value_map.numpy())
                # plt.show()
                col_max, col_max_indice = value_map.max(dim=0)
                max_col_max, max_col_max_indice = col_max.max(dim=0)
                x = max_col_max_indice.item()
                y = col_max_indice[x].item()
                x = x/40.0*8.0
                y = y/25.0*5.0
        else:
            value_map = torch.randn(25, 40).double().to(device)
            value_map *= self.grid
            # plt.imshow(value_map.numpy())
            # plt.show()
            col_max, col_max_indice = value_map.max(0)
            max_col_max, max_col_max_indice = col_max.max(0)
            x = max_col_max_indice.item()
            y = col_max_indice[x].item()
            x = x/40.0*8.0
            y = y/25.0*5.0
            #x, y = random.random()*8.0, random.random()*5.0

        self.target = (x, y)
        try:
            self.move.setGoal(pos, self.target)
        except:
            return action
        self.steps_done += 1

        action = self.move.moveTo(pos, vel, angle, angular, action)
        return action

    def push(self, next_state, reward):
        device = self.device
        target = torch.tensor(self.target, device=device).double()
        next_state = torch.tensor(next_state).to(device).unsqueeze(0).double()
        reward = torch.tensor([reward], device=device).double()
        self.memory.push(self.state, target, next_state, reward)

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
        state_action_values = self.policy_net(state_batch)
        state_action_values = state_action_values.reshape(
            [BATCH_SIZE, -1]).max(dim=1)[0]

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
            next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values)

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
