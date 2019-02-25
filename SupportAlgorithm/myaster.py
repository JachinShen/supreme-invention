# Copyright (c) 2008 Mikael Lind
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from Astar import astar, parse_grid
import random

DUNGEON = """
    #################
                    #
                    #           ###########
                    #                     #
#############       #                     #
#                   #                     #
#                   #                     #
#          ###################            #
#                            #            #
#                            #            #
#                            #       #    #
#                #############       #    #
#                                    #
###############                      #          #
                                     #          #
                                     #          #
                                     #          #
                           ######################
"""

HEIGHT, WIDTH = 22, 79
MAX_LIMIT = HEIGHT * WIDTH
LIMIT = MAX_LIMIT // 2
DEBUG = False
COLOR = True


class Engine(object):

    def __init__(self, grid):
        self.grid = grid
        self.y = random.randrange(self.grid.height)
        self.x = random.randrange(self.grid.width)
        self.goal = (random.randrange(self.grid.height),
                     random.randrange(self.grid.width))
        self.limit = LIMIT
        self.nodes = {}
        self.path = []

    def update_path(self):
        self.path = astar(DUNGEON, (self.y, self.x), 0, self.goal, self.limit, )


if __name__ == '__main__':
    grid = parse_grid(DUNGEON, WIDTH, HEIGHT)
    engine = Engine(grid)
    print('start from (', engine.y, engine.x, ')')
    engine.update_path()
    print(engine.path)
    print('Goal is ', engine.goal)
