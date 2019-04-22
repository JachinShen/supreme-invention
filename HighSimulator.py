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

DIRECTION_LEFT = 0
DIRECTION_UP = 1
DIRECTION_RIGHT = 2
DIRECTION_DOWN = 3


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
        self.health = 2000
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


class HighSimulator():
    def __init__(self):
        self.robots = []
        self.map = np.zeros((7, 4))  # 8x5
        self.map[3, 0] = MAP_TYPE_SUPPLY_BLUE
        self.map[3, 3] = MAP_TYPE_SUPPLY_RED
        self.map[1, 2] = MAP_TYPE_BUFF_BLUE
        self.map[5, 1] = MAP_TYPE_BUFF_RED
        self.map_robots = np.zeros((7, 4, 2), dtype=int)  # 8x5

        self.available_action = [
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

    def reset(self):
        self.robots = [
            Robot([0, 3], DIRECTION_RIGHT, side=ROBOT_BLUE),
            Robot([6, 0], DIRECTION_RIGHT, side=ROBOT_RED),
        ]
        for robot in self.robots:
            x, y = robot.pos
            self.map_robots[x, y] = robot.side

    def step(self):
        self.updateMapRobot()
        ### detect ###
        for robot in self.robots:
            x, y = robot.pos
            direction = robot.direction
            robot.detect = -1
            opposide = oppositeRobot(robot.side)
            while direction in self.available_action[x][y]:
                x, y = robot.getNextPos(direction, [x, y])
                if self.map_robots[x, y, 0] == opposide:
                    robot.detect = self.map_robots[x, y, 1]

        ### shoot ###
        for robot in self.robots:
            if robot.detect >= 0:
                self.robots[robot.detect].health -= 50 * 4
                robot.bullets -= 5

        ### move ###
        for robot in self.robots:
            x, y = robot.pos
            avail_a = self.available_action[x][y]
            for a in avail_a:
                x, y = robot.getNextPos(a)
                if (self.map_robots[x, y, 0] != MAP_TYPE_NORMAL or
                    (self.map[x, y] == MAP_TYPE_SUPPLY_BLUE and robot.side == ROBOT_RED) or
                        (self.map[x, y] == MAP_TYPE_SUPPLY_RED and robot.side == ROBOT_BLUE)):
                    avail_a.remove(a)
            a = robot.chooseAction(avail_a)
            robot.move(a)

        ### show state ###
        for robot in self.robots:
            print(robot.health)

    def updateMapRobot(self):
        self.map_robots = np.zeros((7, 4, 2), dtype=int)
        for i, robot in enumerate(self.robots):
            x, y = robot.pos
            self.map_robots[x, y, 0] = robot.side
            self.map_robots[x, y, 1] = i

    def show(self):
        self.updateMapRobot()
        draw = self.map + self.map_robots[:, :, 0]
        plt.imshow(draw.T, origin="buttom",
                   extent=(0, 7, 0, 4), cmap="RdBu", vmin=-5, vmax=5)


if __name__ == "__main__":
    simulator = HighSimulator()
    simulator.reset()
    for i in range(2*60):
        simulator.step()
        simulator.show()
        plt.pause(0.01)
