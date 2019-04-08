import math
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(".")
from Referee.ICRAMap import BORDER_BOX, BORDER_POS, OBSTACLE_BOX, OBSTACLE_POS


def calc_repulsive_potential(x, y, ox, oy, rr):
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(ox):
        #d = np.hypot(x - ox[i], y - oy[i])
        d = math.sqrt((x - ox[i])**2 + (y - oy[i])**2)
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    #dq = np.hypot(x - ox[minid], y - oy[minid])
    d = math.sqrt((x - ox[minid])**2 + (y - oy[minid])**2)
    return d
    #return np.array([ox[minid], oy[minid]])

MARGIN = 0
width = 80
height = 50
ox = []
oy = []
for (x, y), (w, h) in zip(BORDER_POS+OBSTACLE_POS, BORDER_BOX+OBSTACLE_BOX):
    for i in np.arange(x-w, x+w, 0.1):
        for j in np.arange(y-h, y+h, 0.1):
            ox.append(i)
            oy.append(j)
ob = np.zeros([height, width])
for ix in range(width):
    x = ix / (width/8)
    for iy in range(height):
        y = iy / (height/5)
        ob[iy, ix] = calc_repulsive_potential(x, y, ox, oy, 0.25)
expanded_grid = np.zeros([height+2*MARGIN, width+2*MARGIN])
#expanded_grid[MARGIN:-MARGIN, MARGIN:-MARGIN] = ob
expanded_grid[:,:] = ob
print(expanded_grid.shape)
plt.imshow(expanded_grid)
plt.show()
np.save("ob.npy", expanded_grid)
