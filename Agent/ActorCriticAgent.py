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

from Agent.Actor import Actor
from Agent.Critic import Critic
from util.Grid import Map, MARGIN

BATCH_SIZE = 512
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50

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

    def sample(self, batch_size, is_test=False):
        if is_test:
            return self.memory[0:batch_size]
        else:
            return random.sample(self.memory[:], batch_size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self,index):
        return self.memory[index]


class ActorCriticAgent():
    def __init__(self):
        # if gpu is to be used
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.actor = Actor().to(device).double()
        self.critic = Critic().to(device).double()

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=5e-4)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=5e-4)
        self.memory = ReplayMemory(100000)

        self.steps_done = 0

        self.scale = 20.0
        self.map_width = int(8*self.scale)
        self.map_height = int(5*self.scale)
        icra_map = Map(self.map_width, self.map_height)
        grid = icra_map.getGrid()
        self.obs_map = torch.from_numpy(1-grid).to(device)
        self.whole_map = torch.from_numpy(np.load("ob.npy")).to(device)
        #self.whole_map = torch.from_numpy(1-grid).to(device)

        self.view_width, self.view_height = 0.8, 0.5 # m(half)
        self.grid_width, self.grid_height = int(self.map_width*(self.view_width*2/8)), int(self.map_height*(self.view_height*2/5))

        x, y = np.mgrid[-self.view_height:self.view_height:1/self.scale,
            -self.view_width:self.view_width:1/self.scale]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        self.pos = pos
        rv = multivariate_normal([0.0, -0.0], [[0.001, 0.0], [0.0, 0.001]])
        gaussian = torch.from_numpy(rv.pdf(pos)).to(device).double()
        #self.gaussian = gaussian.unsqueeze(0).unsqueeze(0).repeat(BATCH_SIZE,1,1,1)
        self.gaussian = gaussian
        self.whole_rand = torch.rand(self.grid_height, self.grid_width).double().to(device)
        self.predicted_value = None
        self.value_map = torch.zeros(1, 1, self.grid_height, self.grid_width).double().to(device)
        #plt.imshow(self.gaussian, vmin=0, cmap="GnBu")
        #plt.show()

    def perprocess_state(self, state):
        device = self.device
        p = state[:2]
        e_p = state[-2:]
        left, bottom = p[0]-self.view_width, p[1]-self.view_height
        left, bottom = int(left*self.scale), int(bottom*self.scale)
        left += MARGIN
        bottom += MARGIN
        right = left + self.grid_width
        top = bottom + self.grid_height
        self.window = (left, right, bottom, top)
        window_map = (self.whole_map[bottom:top, left:right])
        enemy_map = (torch.zeros((self.grid_height), (self.grid_width)).to(device).double())
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
        if False:
            plt.cla()
            plt.xlim(0,self.grid_width-1)
            plt.ylim(0,self.grid_height-1)
            plt.imshow(enemy_map, vmin=0, cmap="GnBu")
            plt.pause(0.1)
        window_map = window_map.unsqueeze(0).unsqueeze(1)
        enemy_map = enemy_map.unsqueeze(0).unsqueeze(1)
        state_map = torch.cat([window_map, enemy_map, self.value_map], dim=1) # 1, 3, h, w
        return state_map

    def select_action(self, state_map, mode):
        device = self.device
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        #pos = (state[0], state[1])
        #vel = (state[2], state[3])
        #angle = state[4]
        #angular = state[5]
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        left, right, bottom, top = self.window
        self.state_obs = state_map

        #if is_test or sample > eps_threshold:
        if mode == "max_probability":
            with torch.no_grad():
                #self.policy_net.eval()
                self.value_map = self.actor(state_map)
                value_map = self.value_map[0][0]
            value_map *= self.obs_map[bottom:top, left:right]
            h, w = value_map.shape
            if True:
                plt.cla()
                plt.xlim(0,self.grid_width-1)
                plt.ylim(0,self.grid_height-1)
                plt.imshow(value_map.cpu().numpy().reshape([h,w]), cmap="coolwarm", vmin=0, )
                #plt.imshow(semantic[0][10], cmap="GnBu", vmin=0, )
                plt.pause(0.00001)
            col_max, col_max_indice = value_map.max(dim=0)
            max_col_max, max_col_max_indice = col_max.max(dim=0)
            x = max_col_max_indice.item()
            y = col_max_indice[x].item()
            self.local_goal = [x, y]
            x += (left-MARGIN)
            y += (bottom-MARGIN)
            x = x/self.map_width*8.0
            y = y/self.map_height*5.0
            self.global_goal = [x, y]
        #else:
        #elif sample > eps_threshold:
        elif mode == "sample":
            with torch.no_grad():
                #self.policy_net.eval()
                self.value_map = self.actor(state_map)
                value_map = self.value_map[0][0]
            value_map *= self.obs_map[bottom:top, left:right]
            h, w = value_map.shape
            value_map = value_map.reshape([w*h])
            value_map /= value_map.sum()
            value_map = value_map.cpu().numpy()
            selected = np.random.choice(range(w*h), p=value_map)
            y = selected // w
            x = selected % w
            self.local_goal = [x, y]
            x += (left-MARGIN)
            y += (bottom-MARGIN)
            x = x/self.map_width*8.0
            y = y/self.map_height*5.0
            self.global_goal = [x, y]
        elif mode == "random":
            self.whole_rand = self.whole_rand*0.9 + torch.rand(top-bottom, right-left).double().to(device)*0.1
            value_map = self.whole_rand
            value_map -= self.gaussian[0][0]
            #plt.imshow(value_map.numpy())
            #plt.show()
            value_map += self.obs_map[bottom:top, left:right]
            col_max, col_max_indice = value_map.max(dim=0)
            max_col_max, max_col_max_indice = col_max.max(dim=0)
            x = max_col_max_indice.item()
            y = col_max_indice[x].item()
            self.local_goal = [x, y]
            x += (left-MARGIN)
            y += (bottom-MARGIN)
            x = x/self.map_width*8.0
            y = y/self.map_height*5.0
            self.global_goal = [x, y]
        
        return [x, y]

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
        transitions = self.memory.sample(BATCH_SIZE, is_test=False)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                #batch.next_state)), device=device, dtype=torch.uint8)
        #non_final_next_states = torch.cat([s for s in batch.next_state
                                           #if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        ### Critic ###
        value_eval = self.critic(state_batch)
        next_value_eval = self.critic(next_state_batch)
        td_error = reward_batch / 100.0 + GAMMA*next_value_eval - value_eval
        loss = torch.sum(td_error ** 2)
        self.optimizer_critic.zero_grad()
        loss.backward()
        for param in self.critic.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer_critic.step()

        ### Actor ###
        state_batch = Variable(state_batch, requires_grad=True)
        state_action_prob = self.actor(state_batch) # batch, 1, 10, 16
        action_batch = action_batch[:,1:]*state_action_prob.size(3)+action_batch[:,:1]
        state_action_prob = state_action_prob.reshape([BATCH_SIZE, -1])
        state_action_prob = state_action_prob.gather(1, action_batch)
        log_prob = torch.log(state_action_prob)
        exp_v = torch.mean(log_prob * td_error.detach())
        loss = -exp_v
        self.optimizer_actor.zero_grad()
        loss.backward()
        for param in self.actor.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer_actor.step()

        return loss.item()

    def test_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        device = self.device
        transitions = self.memory.sample(BATCH_SIZE, is_test=True)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        with torch.no_grad():
            ### Critic ###
            value_eval = self.critic(state_batch)
            next_value_eval = self.critic(next_state_batch)
            td_error = reward_batch / 100.0 + GAMMA*next_value_eval - value_eval

            ### Actor ###
            #state_batch = Variable(state_batch, requires_grad=True)
            state_action_prob = self.actor(state_batch) # batch, 1, 10, 16
            action_batch = action_batch[:,1:]*state_action_prob.size(3)+action_batch[:,:1]
            state_action_prob = state_action_prob.reshape([BATCH_SIZE, -1])
            state_action_prob = state_action_prob.gather(1, action_batch)
            log_prob = torch.log(state_action_prob)
            exp_v = torch.mean(log_prob * td_error.detach())
            loss = -exp_v

        return loss.item()

    def save(self):
        torch.save(self.actor.state_dict(), "Actor.model")
        torch.save(self.critic.state_dict(), "Critic.model")
    
    def save_memory(self, file_name):
        torch.save(self.memory, file_name)

    def load(self):
        self.actor.load_state_dict(torch.load(
            "Actor.model", map_location=self.device))
        self.critic.load_state_dict(torch.load(
            "Critic.model", map_location=self.device))

    def load_memory(self, file_name):
        self.memory = torch.load(file_name)
