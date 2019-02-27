import sys
import math
import numpy as np

import sys

sys.path.append("..")
from SupportAlgorithm.Astar import astar, pathprocess
from util.Grid import map2grid, view_path, grid2world, world2grid

WIDTH = 100
HEIGHT = 60
MINBIAS = 0.2

MAXVELOCITY = 0.8
ACC = 1.0


class MoveAction:
    def __init__(self, target, observation):
        self.index = 0
        DUNGEON = map2grid(WIDTH, HEIGHT)
        self.target_grid = world2grid(target)
        self.selfpos_gird = world2grid(observation["pos"])
        self.path = pathprocess(astar(DUNGEON, WIDTH, HEIGHT, self.selfpos_gird, 0, self.target_grid))
        self.tonext = 1000
        self.velocity = observation["velocity"]

    def MoveTo(self, observation, action):
        selfpos = observation["pos"]
        self.velocity = observation["velocity"]
        if self.index + 1 < self.path.__len__():
            nexttarget = grid2world(self.path[self.index + 1])
            self.tonext = self.dist(selfpos, nexttarget)
            if self.tonext < MINBIAS:
                self.index += 1
            else:
                action = self.MoveSubTo(nexttarget, selfpos, self.velocity, action)

        return action

    def MoveSubTo(self, target, selfpos, velocity, action):
        distance = np.sqrt(np.square(target.x - selfpos.x) + np.square(target.y - selfpos.y))
        delta = target - selfpos
        decelerate = 0.5 * np.square(MAXVELOCITY) / ACC

        if distance < MINBIAS or abs(velocity.x) >= MAXVELOCITY or abs(velocity.y) >= MAXVELOCITY:
            action[0] = +0.0
            action[2] = +0.0
        else:
            if distance < decelerate and abs(velocity.x) >= MAXVELOCITY:
                if velocity.x > 0:
                    action[0] = -ACC
                elif velocity.x < 0:
                    action[0] = ACC
            if distance < decelerate and abs(velocity.y) >= MAXVELOCITY:
                if velocity.y > 0:
                    action[2] = -ACC
                elif velocity.y < 0:
                    action[2] = ACC
            if delta.x > MINBIAS:
                action[0] = +ACC
            elif delta.x < -MINBIAS:
                action[0] = -ACC
            if delta.y > MINBIAS:
                action[2] = -ACC
            elif delta.y < -MINBIAS:
                action[2] = +ACC

        return action

    def dist(self, selfpos, target):
        distance = np.sqrt(np.square(target.x - selfpos.x) + np.square(target.y - selfpos.y))
        return distance


'''
The cood of Box2Dworld:
 /\ y     
 ||
 ||
 ||
 ||
 ||
 ====================>
                   x


 The cood of Grid map:
 ====================>x
||
||
||
||
||
\/ y
So it is up-down flipped

 #####################################################################################              
 #####################################################################################              
########################################################################################            
########################################################################################            
########################################################################################            
########################################################################################            
########################################################################################            
######                            #######                                        #######            
######                            #######                                        #######            
######                            #######                                        #######            
######                            #######                                        #######            
######                            #######                                        #######            
######                            #######                  ###############       #######            
######                            #######                  ###############       #######            
######                            #######                  ###############       #######            
######                            #######                  ###############       #######            
######         ########           #######                  ###############       #######            
######         ########                                    ###############       #######            
######         ########                                    ###############       #######            
######         ########                                    ###############       #######            
######         ########                                                          #######            
######       : ########                                                          #######            
######       : ########                                                          #######            
######       : ########                                                          #######            
######       : ########                                                          #######            
######       : ########                                                          #######            
######       : ########             ###############                              #######            
######       : ########             ###############                              #######            
######       : ########             ###############              #######         #######            
######       : ########             ###############              #######         #######            
######       : ########             ###############              #######         #######            
######        :                     ###############              #######         #######            
######         :                    ###############              #######         #######            
######          :                                                #######         #######            
######           :                                               #######         #######            
######            :                                              #######         #######            
######             :                                             #######         #######            
######              :                                            #######         #######            
######               :                                           #######         #######            
######                :::::                                      #######         #######            
######       ##############:                                     #######         #######            
######       ###############:::::::::::::::::::::::::            #######         #######            
######       ###############                  #######:           #######         #######            
######       ###############                  ########:                          #######            
######       ###############                  ######## :                         #######            
######       ###############                  ########  :                        #######            
######       ###############                  ########   :                       #######            
######                                        ########    :                      #######            
######                                        ########     :                     #######            
######                                        ########     :                     #######            
######                                        ########     :                     #######            
######                                        ########                           #######            
########################################################################################            
########################################################################################            
########################################################################################            
########################################################################################            
########################################################################################            
 #####################################################################################              
 #####################################################################################     
'''
