import sys
import Box2D
import random
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(".")
from Referee.ICRAMap import BORDER_POS, BORDER_BOX
from Objects.Robot import ROBOT_SIZE,SIZE

#BORDER_POS = [(-0.1, 2.5), (4, -0.1), (4, 5.1), (8.1, 2.5), (1.525, 1.9), (3.375, 0.5), (6.475, 3.1), (4.625, 4.5),
              #(1.7, 3.875), (4, 2.5), (6.3, 1.125)]
#BORDER_BOX = [(0.1, 2.5), (4, 0.1), (4, 0.1), (0.1, 2.5), (0.125, 0.5), (0.125, 0.5), (0.125, 0.5), (0.125, 0.5),
              #(0.5, 0.125), (0.5, 0.125), (0.5, 0.125)]  # Half of the weight and height

MAXSCALE = 20

BIAS = 0.5
BODYSIZE = SIZE*ROBOT_SIZE*1.4

class Cell(object):

    def __init__(self, char):
        self.char = char
        self.tag = 0
        self.index = 0
        self.neighbors = None


class Grid(object):

    def __init__(self, cells):
        self.height, self.width = len(cells), len(cells[0])
        self.cells = cells

    def __contains__(self, pos):
        y, x = pos
        return 0 <= y < self.height and 0 <= x < self.width

    def __getitem__(self, pos):
        y, x = pos
        return self.cells[y][x]

    def neighbors(self, y, x):
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1),
                       (1, 0), (1, 1)):
            if (y + dy, x + dx) in self:
                yield y + dy, x + dx

class Map():
    def __init__(self, width, height):
        grid = np.zeros([height, width])
        scale_x = width/8.0
        scale_y = height/5.0
        index_x = []
        index_y = []
        self.width = width
        self.height = height
        self.scale_x = scale_x
        self.scale_y = scale_y

        for (x, y), (w, h) in zip(BORDER_POS, BORDER_BOX):
            for idx in range(int((x-w-0.3)*scale_x), int((x+w+0.6)*scale_x)):
                if idx < 0 or idx >= width:
                    continue
                for idy in range(int((y-h-0.3)*scale_y), int((y+h+0.6)*scale_y)):
                    if idy < 0 or idy >= height:
                        continue
                    index_x.append(idx)
                    index_y.append(idy)

        index_x = np.array(index_x)
        index_y = np.array(index_y)
        #index_x = np.clip(index_x, 0, width-1)
        #index_y = np.clip(index_y, 0, height-1)

        grid[index_y, index_x] = 1
        self.grid = grid

    def getGrid(self):
        return self.grid

    def getXYIndex(self, x, y):
        return x*self.scale_x, y*self.scale_y



def map2grid(width, height):
    str = ''

    for i in range(height):
        str += '\n'
        for j in range(width):
            str += ' '

    for k in range(BORDER_POS.__len__()):
        hidx = (int(MAXSCALE * BIAS + MAXSCALE * BORDER_POS[k][1] - MAXSCALE * BORDER_BOX[k][1] - MAXSCALE * BODYSIZE),
                int(MAXSCALE * BIAS + MAXSCALE * BORDER_POS[k][1] + MAXSCALE * BORDER_BOX[k][1] + MAXSCALE * BODYSIZE))
        widx = (int(MAXSCALE * BIAS + MAXSCALE * BORDER_POS[k][0] - MAXSCALE * BORDER_BOX[k][0] - MAXSCALE * BODYSIZE),
                int(MAXSCALE * BIAS + MAXSCALE * BORDER_POS[k][0] + MAXSCALE * BORDER_BOX[k][0] + MAXSCALE * BODYSIZE))
        for i in range(hidx[0], hidx[1]):
            for j in range(max(widx[0], 1), widx[1]):
                mylen = i * (width + 1) + j
                str = str[:mylen] + '#' + str[mylen + 1:]

    return str


def parse_grid(grid_str, width, height):
    # Split the grid string into lines.
    lines = [line.rstrip() for line in grid_str.splitlines()[1:]]

    # Pad the top and bottom.
    top = 0#(height - len(lines)) // 2
    bottom =height #(height - len(lines) + 1) // 2
    lines = ([''] * top + lines + [''] * bottom)[:height]

    # Pad the left and right sides.
    max_len = max(len(line) for line in lines)
    left = 0#(width - max_len) // 2
    lines = [' ' * left + line.ljust(width - left)[:width - left]
             for line in lines]

    # Create the grid.
    cells = [[Cell(char) for char in line] for line in lines]
    return Grid(cells)

def view_grid(grid):
    # Update the grid view.
    str = ''
    for y, line in enumerate(grid.cells):
        for x, cell in enumerate(line):
            char = cell.char
            if (char == '#'):
                str += '#'
            else:
                str += ' '
        str += '\n'
    return str


def view_path(str, path, width):
    for i in range(path.__len__()):
        mylen = path[i][0] * (width + 1) + path[i][1] + 1
        str = str[:mylen] + ':' + str[mylen + 1:]

    return str

BIASX = 3
BIASY = 0
def grid2world(path):
    cood = Box2D.b2Vec2((float(path[1] + BIASX)) / MAXSCALE - BIAS,
                        (float(path[0]) + BIASY) / MAXSCALE - BIAS)
    return cood


def world2grid(cood):
    path = (int(MAXSCALE * BIAS + MAXSCALE * cood.y) - BIASY,
            int(MAXSCALE * BIAS + MAXSCALE * cood.x) - BIASX)

    return path


if __name__ == '__main__':

    width = 160
    height = 100
    grid = map2array(width, height)
    plt.imshow(grid)
    plt.show()
    '''
    str = map2grid(width, height)
    print(str)
    mylen = str.__len__()
    grid = parse_grid(str, width, height)
    str=view_grid(grid)
    print(str)
    '''
