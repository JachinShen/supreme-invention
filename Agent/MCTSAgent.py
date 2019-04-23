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
ACTION_STAY = 4
ACTION_STRING = ["Left", "Up", "Right", "Down", "Stay"]

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

GRID_POS = [
    [[50, 50], [55, 140], [50, 235], [50, 320]],
    [[155, 50], [155, 140], [135, 235], [122, 320]],
    [[190, 50], [200, 130], [200, 235], [200, 320]],
    [[290, 35], [285, 125], [285, 235], [280, 330]],
    [[370, 40], [370, 130], [375, 225], [385, 320]],
    [[450, 40], [450, 130], [410, 225], [430, 325]],
    [[535, 50], [530, 135], [520, 230], [520, 310]],
]


def oppositeRobot(side):
    if side == ROBOT_BLUE:
        return ROBOT_RED
    elif side == ROBOT_RED:
        return ROBOT_BLUE
    else:
        print("Wrong input!")
        return side


def discreteRad(r):
    assert(-np.pi <= r <= np.pi)
    d = r / np.pi * 180.0
    if 180 >= d >= 135:
        return DIRECTION_LEFT
    elif 135 >= d >= 45:
        return DIRECTION_UP
    elif 45 >= d >= -45:
        return DIRECTION_RIGHT
    elif -45 >= d >= -135:
        return DIRECTION_DOWN
    elif -135 >= d >= -180:
        return DIRECTION_LEFT
    else:
        print("Error!")
        return None


def revertXYGrid(pos):
    x, y = pos
    x, y = x*574.0/8.0, y*360.0/5.0
    x, y = int(x), int(y)
    #print(x, y)
    distance = np.zeros((7, 4))
    for i in range(7):
        for j in range(4):
            grid_pos = GRID_POS[i][j]
            distance[i, j] = (grid_pos[0]-x)**2 + (grid_pos[1]-y)**2

    min_id = int(np.argmin(distance))
    #print("Revert grid x: {}, y: {}".format(min_id // 4, min_id % 4))
    return min_id // 4, min_id % 4


class Robot():
    def __init__(self, pos, direction, side):
        self.pos = pos
        self.direction = direction
        self.side = side
        self.health = 500
        self.buff_left_time = 0
        self.bullets = 1000
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
        elif action == ACTION_STAY:
            pass
        return [x, y]

    def move(self, action):
        self.pos = self.getNextPos(action)
        if self.buff_left_time > 0:
            self.buff_left_time -= 1

    def __str__(self):
        return "Robot pos: {}, health: {}".format(self.pos, self.health)

    def __repr__(self):
        return str(self)


class NaughtsAndCrossesState():
    def __init__(self):
        self.robots = [
            Robot([0, 0], DIRECTION_RIGHT, side=ROBOT_RED),
            Robot([6, 3], DIRECTION_RIGHT, side=ROBOT_BLUE),
        ]
        self.currentPlayer = 0
        self.index = 0
        self.buff_chance = 1

    def getPossibleActions(self):
        x, y = self.robots[self.currentPlayer].pos
        possibleActions = AVAILABLE_ACTION[x][y]
        return possibleActions

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.index = self.index + 1

        robot = newState.robots[newState.currentPlayer]
        ### detect ###
        direction = robot.direction
        robot.detect = -1
        opposide = newState.robots[1-newState.currentPlayer]
        x, y = robot.pos
        while direction in AVAILABLE_ACTION[x][y]:
            x, y = robot.getNextPos(direction, [x, y])
            if [x, y] == opposide.pos:
                robot.detect = 1-newState.currentPlayer
                break
        ### shoot ###
        if robot.detect >= 0:
            if newState.robots[robot.detect].buff_left_time > 0:
                #print("Buff, health: {}".format(robot.health))
                newState.robots[robot.detect].health -= 25 * 4
            else:
                newState.robots[robot.detect].health -= 50 * 4
            robot.bullets -= 5
        ### move ###
        robot.move(action)
        robot.direction = action
        x, y = robot.pos
        if (newState.buff_chance > 0 and (
            (robot.side == ROBOT_BLUE and FIELD_MAP[x][y] == MAP_TYPE_BUFF_BLUE) or
            (robot.side == ROBOT_RED and FIELD_MAP[x][y] == MAP_TYPE_BUFF_RED)
        )):
            print("Index: {}".format(self.index))
            newState.buff_chance = 0
            robot.buff_left_time = 30
            #newState.robots[1].health = -1
            #robot.health = 100000
        newState.robots[newState.currentPlayer] = robot
        newState.currentPlayer = (self.currentPlayer + 1) % 2

        return newState

    def isTerminal(self):
        if self.index > 7 * 60:
            return True
        for robot in self.robots:
            if robot.health <= 0:
                return True
        return False

    def getReward(self):
        # if self.isTerminal():
        #return self.robots[self.currentPlayer].health - self.robots[1-self.currentPlayer].health
        return self.robots[0].health - self.robots[1].health
        # else:
        # return False


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


class MCTSAgent():
    def __init__(self):
        self.state = NaughtsAndCrossesState()
        self.mcts = mcts(timeLimit=100)

    def select_action(self, state):
        self.state.currentPlayer = 0
        x, y = state["robot_0"]["pos"]
        #x, y = int(x/8.0*7.0), int(y/5.0*4.0)
        self.state.robots[0].pos = revertXYGrid([x, y])
        self.state.robots[0].direction = discreteRad(state["robot_0"]["angle"])
        self.state.robots[0].health = state["robot_0"]["health"]
        x, y = state["robot_1"]["pos"]
        #x, y = int(x/8.0*7.0), int(y/5.0*4.0)
        self.state.robots[1].pos = revertXYGrid([x, y])
        #self.state.robots[1].pos = [x, y]
        self.state.robots[1].direction = discreteRad(state["robot_1"]["angle"])
        self.state.robots[1].health = state["robot_1"]["health"]
        action = self.mcts.search(initialState=self.state)
        #print(ACTION_STRING[action], state.getReward())
        self.state = self.state.takeAction(action)
        # print(self.state.index)
        x, y = self.state.robots[0].pos
        #print(x, y)
        x, y = GRID_POS[x][y]
        x, y = x/574.0*8.0, y/360.0*5.0
        return [x, y]

    def reset(self):
        self.state.index = 0
