from __future__ import division

from copy import deepcopy
from mcts import mcts
from functools import reduce
import operator
import numpy as np
import matplotlib.pyplot as plt

MAP_TYPE_NORMAL = 0
MAP_TYPE_SUPPLY_BLUE = 1
MAP_TYPE_SUPPLY_RED = -1
MAP_TYPE_BUFF_BLUE = 2
MAP_TYPE_BUFF_RED = -2

ROBOT_BLUE = 3
ROBOT_RED = -3

ACTION_LEFT = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3
ACTION_STRING = ["Left", "Up", "Right", "Down"]

DIRECTION_LEFT = 0
DIRECTION_UP = 1
DIRECTION_RIGHT = 2
DIRECTION_DOWN = 3

AVAILABLE_ACTION = [
    [
        [ACTION_UP, ACTION_RIGHT],
        [ACTION_UP, ACTION_DOWN],
        [ACTION_UP, ACTION_RIGHT, ACTION_DOWN],
        [ACTION_RIGHT, ACTION_DOWN],
    ], [
        [ACTION_LEFT, ACTION_UP, ACTION_RIGHT],
        [ACTION_UP, ACTION_RIGHT, ACTION_DOWN],
        [ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN],
        [ACTION_LEFT, ACTION_RIGHT],
    ], [
        [ACTION_LEFT, ACTION_UP],
        [ACTION_LEFT, ACTION_UP, ACTION_RIGHT, ACTION_DOWN],
        [ACTION_LEFT, ACTION_UP, ACTION_RIGHT, ACTION_DOWN],
        [ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN],
    ], [
        [ACTION_UP, ACTION_RIGHT],
        [ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN],
        [ACTION_LEFT, ACTION_UP, ACTION_RIGHT],
        [ACTION_LEFT, ACTION_DOWN],
    ], [
        [ACTION_LEFT, ACTION_UP, ACTION_RIGHT],
        [ACTION_LEFT, ACTION_UP, ACTION_RIGHT, ACTION_DOWN],
        [ACTION_LEFT, ACTION_UP, ACTION_RIGHT, ACTION_DOWN],
        [ACTION_RIGHT, ACTION_DOWN],
    ], [
        [ACTION_LEFT, ACTION_RIGHT],
        [ACTION_LEFT, ACTION_UP, ACTION_RIGHT],
        [ACTION_LEFT, ACTION_UP, ACTION_DOWN],
        [ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN],
    ], [
        [ACTION_LEFT, ACTION_UP],
        [ACTION_LEFT, ACTION_UP, ACTION_DOWN],
        [ACTION_UP, ACTION_DOWN],
        [ACTION_LEFT, ACTION_DOWN],
    ], ]
FIELD_MAP = np.zeros((7, 4))  # 8x5
FIELD_MAP[3, 0] = MAP_TYPE_SUPPLY_BLUE
FIELD_MAP[3, 3] = MAP_TYPE_SUPPLY_RED
FIELD_MAP[1, 2] = MAP_TYPE_BUFF_BLUE
FIELD_MAP[5, 1] = MAP_TYPE_BUFF_RED

def oppositeRobot(side):
    if side == ROBOT_BLUE:
        return ROBOT_RED
    elif side == ROBOT_RED:
        return ROBOT_BLUE
    else:
        print("Wrong input!")
        return side

class Robot():
    def __init__(self, pos, direction, side):
        self.pos = pos
        self.direction = direction
        self.side = side
        self.health = 500
        self.buff_left_time = 0
        self.bullets = 100
        self.detect = -1

    def chooseAction(self, available_action):
        if self.detect >= 0:
            print("Detect")
        return np.random.choice(available_action)

    def getNextPos(self, action, pos=-1):
        if pos == -1:
            x, y = self.pos
        else:
            x, y = pos
        if action == ACTION_LEFT:
            x -= 1
        elif action == ACTION_UP:
            y += 1
        elif action == ACTION_RIGHT:
            x += 1
        elif action == ACTION_DOWN:
            y -= 1
        return [x, y]

    def move(self, action):
        self.pos = self.getNextPos(action)


class NaughtsAndCrossesState():
    def __init__(self):
        self.robots = [
            Robot([0, 3], DIRECTION_RIGHT, side=ROBOT_BLUE),
            Robot([6, 0], DIRECTION_RIGHT, side=ROBOT_RED),
        ]
        self.currentPlayer = 0
        self.index = 0

    def getPossibleActions(self):
        x, y = self.robots[self.currentPlayer].pos
        possibleActions = AVAILABLE_ACTION[x][y]
        return possibleActions

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.index = self.index + 1

        robot = newState.robots[newState.currentPlayer]
        ### detect ###
        x, y = robot.pos
        direction = robot.direction
        robot.detect = -1
        opposide = newState.robots[1-newState.currentPlayer]
        while direction in AVAILABLE_ACTION[x][y]:
            x, y = robot.getNextPos(direction, [x, y])
            if [x, y] == opposide.pos:
                robot.detect = 1-newState.currentPlayer
                break
        ### shoot ###
        if robot.detect >= 0:
            newState.robots[robot.detect].health -= 50 * 4
            robot.bullets -= 5
        ### move ###
        robot.move(action)
        robot.direction = action
        newState.currentPlayer = (self.currentPlayer + 1) % 2

        return newState

    def isTerminal(self):
        if self.index > 2 * 60:
            return True
        for robot in self.robots:
            if robot.health < 0:
                return True
        return False

    def getReward(self):
        #if self.isTerminal():
        return self.robots[self.currentPlayer].health - self.robots[1-self.currentPlayer].health
        #else:
            #return False


class Action():
    def __init__(self, player, action):
        self.player = player
        self.action = action

    def __str__(self):
        return ACTION_STRING[self.action]

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.action == other.action and self.player == other.player

    def __hash__(self):
        return hash((self.action, self.player))


state = NaughtsAndCrossesState()
mcts = mcts(timeLimit=100)
for i in range(2*60):
    action = mcts.search(initialState=state)
    state = state.takeAction(action)
    #state = state.takeAction(np.random.choice(state.getPossibleActions()))
    print(ACTION_STRING[action], state.getReward())
