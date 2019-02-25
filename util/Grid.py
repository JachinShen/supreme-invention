import sys

sys.path.append("..")
import random
from Referee.ICRAMap import BORDER_POS, BORDER_BOX

BORDER_POS = [(-0.1, 2.5), (4, -0.1), (4, 5.1), (8.1, 2.5), (1.525, 1.9), (3.375, 0.5), (6.475, 3.1), (4.625, 4.5),
              (1.7, 3.875), (4, 2.5), (6.3, 1.125)]
BORDER_BOX = [(0.1, 2.5), (4, 0.1), (4, 0.1), (0.1, 2.5), (0.125, 0.5), (0.125, 0.5), (0.125, 0.5), (0.125, 0.5),
              (0.5, 0.125), (0.5, 0.125), (0.5, 0.125)]  # Half of the weight and height


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


def map2grid(width, height):
    str = ''
    MAXSIZE = 10
    BIAS = 0.5
    for i in range(height):
        str += '\n'
        for j in range(width):
            str += ' '

    for k in range(BORDER_POS.__len__()):
        hidx = (int(MAXSIZE * BIAS + MAXSIZE * BORDER_POS[k][1] - MAXSIZE * BORDER_BOX[k][1]),
                int(MAXSIZE * BIAS + MAXSIZE * BORDER_POS[k][1] + MAXSIZE * BORDER_BOX[k][1]))
        widx = (int(MAXSIZE * BIAS + MAXSIZE * BORDER_POS[k][0] - MAXSIZE * BORDER_BOX[k][0]),
                int(MAXSIZE * BIAS + MAXSIZE * BORDER_POS[k][0] + MAXSIZE * BORDER_BOX[k][0]))
        for i in range(hidx[0], hidx[1]):
            for j in range(max(widx[0], 1), widx[1]):
                mylen = i * (width + 1) + j + 1
                str = str[:mylen] + '#' + str[mylen + 1:]

    return str


def parse_grid(grid_str, width, height):
    # Split the grid string into lines.
    lines = [line.rstrip() for line in grid_str.splitlines()[1:]]

    # Pad the top and bottom.
    top = (height - len(lines)) // 2
    bottom = (height - len(lines) + 1) // 2
    lines = ([''] * top + lines + [''] * bottom)[:height]

    # Pad the left and right sides.
    max_len = max(len(line) for line in lines)
    left = (width - max_len) // 2
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

    print(str)
    return str


def view_path(str, path, width):
    for i in range(path.__len__()):
        mylen = path[i][0] * (width + 1) + path[i][1] + 1
        str = str[:mylen] + ':' + str[mylen + 1:]

    return str


if __name__ == '__main__':
    import random

    width = 220
    height = 80
    str = map2grid(width, height)
    mylen = str.__len__()
    grid = parse_grid(str, width, height)
    view_grid(grid)
