import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import cv2
from SupportAlgorithm.GlobalLocalPlanner import GlobalLocalPlanner
from SupportAlgorithm.MoveAction import MoveAction
from SupportAlgorithm.NaiveMove import NaiveMove
from util.Grid import Map

class HandAgent():
    def __init__(self):
        self.avaiable_pos = [
            [0.5, 0.5], [0.5, 2.0], [0.5, 3.0], [0.5, 4.5], # 0 1 2 3 
            [1.5, 0.5], [1.5, 3.0], [1.5, 4.5],             # 4 5 6
            [2.75, 0.5], [2.75, 2.0], [2.75, 3.0], [2.75, 4.5], # 7 8 9 10
            [4.0, 1.75], [4.0, 3.25],                         # 11 12
            [5.25, 0.5], [5.25, 2.0], [5.25, 3.0], [5.25, 4.5], # 13 14 15 16
            [6.5, 0.5], [6.5, 2.0], [6.5, 4.5],             # 17 18 19
            [7.5, 0.5], [7.5, 2.0], [7.5, 3.0], [7.5, 4.5]  # 20 21 22 23
        ]
        self.connected = [
            [1,2,3,4], [0,2,3], [0,1,3,5], [0,1,2,6],
            [0,7], [2,9], [3,10],
            [8,9,10,4], [7,9,10,11], [7,8,10,5,12], [7,8,9],
            [8,14], [9, 15],
            [14,15,16,17], [13,15,16,18,11,11,11,11,11], [13,14,16,12,12,12,12,12], [13,14,15,19],
            [13,20], [14,21], [16, 23],
            [21,22,23,17], [20,22,23,18], [20,21,23], [20,21,22,19]
        ]
        self.path = [
            [5.0, 4.5],
            [5.0, 3.0],
            [4.0, 3.0],
            [3.5, 4.7],
            [0.5, 4.5],
            [0.5, 3.0],
            [2.5, 3.0],
            [4.0, 1.5],
            [7.5, 2.0],
            [7.5, 4.5],
        ]
        self.index = len(self.avaiable_pos)-1
        #self.target = self.path[self.index]
        self.index = random.choice(self.connected[self.index])
        self.target = self.avaiable_pos[self.index]
        self.move = NaiveMove()

    def reset(self, pos):
        #self.index = len(self.avaiable_pos)-1
        self.index = self.avaiable_pos.index(pos)
        self.index = random.choice(self.connected[self.index])
        self.target = self.avaiable_pos[self.index]
        pass

    def select_action(self, state):
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pos = (state[0], state[1])
        vel = (state[2], state[3])
        angle = state[4]
        if state[-1] > 0 and state[-3] > 0:
            pass
            #action[4] = +1.0
            #return action
            #self.target = (state[-4], state[-3])
            #self.move = MoveAction(self.target, pos, vel, angle)
        else:
            action[4] = 0.0
        if ((pos[0]-self.target[0])**2 + (pos[1]-self.target[1])**2 < 0.1):
            self.index = random.choice(self.connected[self.index])
            self.target = self.avaiable_pos[self.index]
            #print(self.target)
            #self.index = (self.index + 1) % len(self.path)
            #self.target = self.path[self.index]

        v, omega = self.move.moveTo(pos, vel, angle, self.target)
        action[0] = v[0]
        action[1] = omega
        action[2] = v[1]
        return action

        return new_action

'''
class HandAgent():
    def __init__(self):
        self.target = (random.random()*8.0, random.random()*5.0)
        self.move = GlobalLocalPlanner()
        self.ctr = 0
        icra_map = Map(40, 25)
        grid = icra_map.getGrid()
        grid = 1 - grid
        self.grid = torch.from_numpy(grid)

    def reset(self):
        self.ctr = 0

    def select_action(self, state):
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pos = (state[0], state[1])
        vel = (state[2], state[3])
        angle = state[4]
        angular = state[5]
        if state[-1] > 0 and state[-3] > 0:
            action[4] = +1.0
        else:
            action[4] = 0.0
        if self.ctr % 5 == 0:
            if state[-1] > 0 and state[-3] > 0:
                self.target = (state[-4], state[-3])
                if self.move.done or ((pos[0]-self.target[0])**2 + (pos[1]-self.target[1])**2 > 4):
                    self.move.setGoal(pos, self.target)
            else:
                if self.move.done or ((pos[0]-self.target[0])**2 + (pos[1]-self.target[1])**2 < 4):
                    value_map = torch.randn(25, 40).double()
                    value_map *= self.grid
                    # plt.imshow(value_map.numpy())
                    # plt.show()
                    col_max, col_max_indice = value_map.max(0)
                    max_col_max, max_col_max_indice = col_max.max(0)
                    x = max_col_max_indice.item()
                    y = col_max_indice[x].item()
                    x = x/40.0*8.0
                    y = y/25.0*5.0
                    self.target = (x, y)
                    #self.target = (7.5, 0.5)
                    #print(pos, self.target)
                    try:
                        self.move.setGoal(pos, self.target)
                    except:
                        return action
        self.ctr += 1

        new_action = self.move.moveTo(pos, vel, angle, angular, action)

        return new_action
'''