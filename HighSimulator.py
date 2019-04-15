import numpy as np
import matplotlib.pyplot as plt

MAP_TYPE_NORMAL = 0
MAP_TYPE_SUPPLY_BLUE = 1
MAP_TYPE_SUPPLY_RED = -1
MAP_TYPE_BUFF_BLUE = 2
MAP_TYPE_BUFF_RED = -2
MAP_TYPE_ROBOT_BLUE = 3
MAP_TYPE_ROBOT_RED = -3

ACTION_LEFT = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3


class Robot():
    def __init__(self, pos, side):
        self.pos = pos
        self.side = side
        self.health = 2000
        self.buff_left_time = 0
        self.bullets = 100

    def chooseAction(self, available_action):
        return np.random.choice(available_action)

    def getNextPos(self, action):
        x, y = self.pos
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
        self.map_robots = np.zeros((7, 4))  # 8x5

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
            Robot([0, 3], side=MAP_TYPE_ROBOT_BLUE),
            Robot([6, 0], side=MAP_TYPE_ROBOT_RED)
        ]
        for robot in self.robots:
            x, y = robot.pos
            self.map_robots[x, y] = robot.side

    def step(self):
        self.updateMapRobot()
        for robot in self.robots:
            x, y = robot.pos
            avail_a = self.available_action[x][y]
            for a in avail_a:
                x, y = robot.getNextPos(a)
                if (self.map_robots[x, y] != MAP_TYPE_NORMAL or
                    (self.map[x, y] == MAP_TYPE_SUPPLY_BLUE and robot.side == MAP_TYPE_ROBOT_RED) or
                        (self.map[x, y] == MAP_TYPE_SUPPLY_RED and robot.side == MAP_TYPE_ROBOT_BLUE)):
                    avail_a.remove(a)
            a = robot.chooseAction(avail_a)
            robot.move(a)

    def updateMapRobot(self):
        self.map_robots = np.zeros((7, 4))
        for robot in self.robots:
            x, y = robot.pos
            self.map_robots[x, y] = robot.side

    def show(self):
        self.updateMapRobot()
        draw = self.map + self.map_robots
        plt.imshow(draw.T,
                   origin="buttom", extent=(0, 7, 0, 4), cmap="RdBu", vmin=-5, vmax=5)
        plt.show()


if __name__ == "__main__":
    simulator = HighSimulator()
    simulator.reset()
    for i in range(10):
        simulator.step()
        simulator.show()
