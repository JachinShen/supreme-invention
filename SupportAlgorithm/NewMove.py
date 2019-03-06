import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from extremitypathfinder.extremitypathfinder import PolygonEnvironment as Environment
sys.path.append(".")

WIDTH = 160
HEIGHT = 100
MINBIAS = 0.03

MAXVELOCITY =1000
ACC = 1

BORDER_POS = [(1.525, 1.9), (6.475, 3.1), 
              (1.7, 3.875), (4, 2.5), (6.3, 1.125)]
BORDER_BOX = [(0.125, 0.5), (0.125, 0.5), 
              (0.5, 0.125), (0.5, 0.125), (0.5, 0.125)]  # Half of the weight and height

ROBOT_SIZE = 0.3

POLYGON_SETTINGS = {
    'edgecolor': 'black',
    'fill': False,
    'linewidth': 1.0,
}

SHOW_PLOTS = True
# parameter
MAX_T = 100.0  # maximum time to the goal [s]
MIN_T = 5.0  # minimum time to the goal[s]

show_animation = True

max_accel = 1.0  # max accel [m/ss]
max_jerk = 0.5  # max jerk [m/sss]

class quinic_polynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):

        # calc coefficient of quinic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[T**3, T**4, T**5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T**2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

        return xt


def mark_points(vertex_iter, **kwargs):
    xs = []
    ys = []
    if type(vertex_iter) == set:
        for v in vertex_iter:
            xs.append(v.coordinates[0])
            ys.append(v.coordinates[1])
    elif type(vertex_iter[0]) == tuple:
        for x, y in vertex_iter:
            xs.append(x)
            ys.append(y)
    else:
        for v in vertex_iter:
            xs.append(v.coordinates[0])
            ys.append(v.coordinates[1])

    plt.scatter(xs, ys, **kwargs)


def draw_edge(v1, v2, c, alpha, **kwargs):
    if type(v1) == tuple:
        x1, y1 = v1
        x2, y2 = v2
    else:
        x1, y1 = v1.coordinates
        x2, y2 = v2.coordinates
    plt.plot([x1, x2], [y1, y2], color=c, alpha=alpha, **kwargs)


def draw_polygon(ax, coords, **kwargs):
    kwargs.update(POLYGON_SETTINGS)
    polygon = Polygon(coords, **kwargs)
    ax.add_patch(polygon)


def draw_boundaries(map, ax):
    # TODO outside light grey
    # TODO fill holes light grey
    draw_polygon(ax, map.boundary_polygon.coordinates)
    for h in map.holes:
        draw_polygon(ax, h.coordinates, facecolor='grey', fill=True)

    mark_points(map.all_vertices, c='black', s=15)
    mark_points(map.all_extremities, c='red', s=50)


def draw_internal_graph(map, ax):
    for start, all_goals in map.graph.get_neighbours():
        for goal in all_goals:
            draw_edge(start, goal, c='red', alpha=0.2, linewidth=2)


def set_limits(map, ax):
    ax.set_xlim((min(map.boundary_polygon.coordinates[:, 0]) - 1, max(map.boundary_polygon.coordinates[:, 0]) + 1))
    ax.set_ylim((min(map.boundary_polygon.coordinates[:, 1]) - 1, max(map.boundary_polygon.coordinates[:, 1]) + 1))


def draw_path(vertex_path):
    # start, path and goal in green
    if vertex_path:
        mark_points(vertex_path, c='g', alpha=0.9, s=50)
        mark_points([vertex_path[0], vertex_path[-1]], c='g', s=100)
        v1 = vertex_path[0]
        for v2 in vertex_path[1:]:
            draw_edge(v1, v2, c='g', alpha=1.0)
            v1 = v2


def draw_prepared_map(map):
    fig, ax = plt.subplots()

    draw_boundaries(map, ax)
    #draw_internal_graph(map, ax)
    set_limits(map, ax)
    if SHOW_PLOTS:
        plt.show()

class NewMove():
    def __init__(self):
        self.environment = Environment()

        # counter clockwise vertex numbering!
        boundary_coordinates = [
            (0+ROBOT_SIZE, 0+ROBOT_SIZE), 
            (3.25-ROBOT_SIZE, 0+ROBOT_SIZE), (3.25-ROBOT_SIZE, 1.0+ROBOT_SIZE), (3.5+ROBOT_SIZE, 1.0+ROBOT_SIZE), (3.5+ROBOT_SIZE, 0+ROBOT_SIZE), 
            (8.0-ROBOT_SIZE, 0+ROBOT_SIZE),
            (8.0-ROBOT_SIZE, 5.0-ROBOT_SIZE), 
            (8.0-3.25+ROBOT_SIZE, 5.0-ROBOT_SIZE), (8.0-3.25+ROBOT_SIZE, 4.0-ROBOT_SIZE), (8.0-3.5-ROBOT_SIZE, 4.0-ROBOT_SIZE), (8.0-3.5-ROBOT_SIZE, 5.0-ROBOT_SIZE), 
            (0+ROBOT_SIZE, 5.0-ROBOT_SIZE)
        ]

        # clockwise numbering!
        list_of_holes = []
        for (x, y), (w, h) in zip(BORDER_POS[:], BORDER_BOX[:]):
            #x, y, w, h = x*10, y*10, w*10, h*10
            list_of_holes.append([
                ((x-w-ROBOT_SIZE), (y-h-ROBOT_SIZE)),
                ((x-w-ROBOT_SIZE), (y+h+ROBOT_SIZE)),
                ((x+w+ROBOT_SIZE), (y+h+ROBOT_SIZE)),
                ((x+w+ROBOT_SIZE), (y-h-ROBOT_SIZE)),
                ])


        self.environment.store(boundary_coordinates, list_of_holes, validate=False)
        self.environment.prepare()


    def plot(self):
        draw_prepared_map(self.environment)
        plt.imshow(self.pmap[:,::-1])
        plt.show()

    def findPath(self, start, goal):
        path, length = self.environment.find_shortest_path(start, goal)
        return path

    def setGoal(self, start, goal):
        self.path = self.findPath(start, goal)
        self.index = 1
        self.next_target = self.path[1]

    def moveTo(self, pos, vel, angle, action):
        if self.distance(pos, self.next_target) < 1:
            self.index += 1
            if self.index < len(self.path):
                self.next_target = self.path[self.index]
            else:
                action[0] = 0.0
                action[2] = 0.0
                return action

        u = np.array([
            self.next_target[0]-pos[0],
            self.next_target[1]-pos[1]])
        u = u.reshape([2, 1])
        mat_angle = np.array([
            [math.cos(angle), math.sin(angle)],
            [math.sin(angle), -math.cos(angle)]])
        v = np.matmul(np.linalg.inv(mat_angle), u) / 10
        print(u, v, angle)
        #action[0] = -v[0][0]
        #action[2] = -v[1][0]
        return action

    def distance(self, p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


if __name__ == "__main__":
    move = NewMove()
    #move.plot()
    start, goal= (0.5, 0.5), (4.5, 0.5)
    print(move.findPath(start, goal))
    move.setGoal(start, goal)
    move.moveTo(start, 0, 0, [])