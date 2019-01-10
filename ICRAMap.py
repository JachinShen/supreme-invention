import numpy as np
import math
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, shape)

# Top-down car dynamics simulation.
#
# Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
# This simulation is a bit more detailed, with wheels rotation.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

SIZE = 0.02
ENGINE_POWER            = 100000000*SIZE*SIZE
WHEEL_MOMENT_OF_INERTIA = 4000*SIZE*SIZE
FRICTION_LIMIT          = 1000000*SIZE*SIZE     # friction ~= mass ~= size^2 (calculated implicitly using density)
WHEEL_R  = 27
WHEEL_W  = 14
WHEELPOS = [
    (-55,+80), (+55,+80),
    (-55,-82), (+55,-82)
    ]
HULL_POLY1 =[
    (-60,+130), (+60,+130),
    (+60,+110), (-60,+110)
    ]
HULL_POLY2 =[
    (-15,+120), (+15,+120),
    (+20, +20), (-20,  20)
    ]
HULL_POLY3 =[
    (+25, +20),
    (+50, -10),
    (+50, -40),
    (+20, -90),
    (-20, -90),
    (-50, -40),
    (-50, -10),
    (-25, +20)
    ]
HULL_POLY4 =[
    (-50,-120), (+50,-120),
    (+50,-90),  (-50,-90)
    ]
WHEEL_COLOR = (0.0,0.0,0.0)
WHEEL_WHITE = (0.3,0.3,0.3)
MUD_COLOR   = (0.4,0.4,0.0)

COLOR_WHITE = (1.0, 1.0, 1.0)

BORDER_POS = [(-0.1, 2.5),(4,-0.1), (4, 5.1), (8.1,2.5),(1.525,1.9),(3.375,0.5),(6.475,3.1),(4.625,4.5),(1.7,3.875),(4,2.5),(6.3, 1.125)]
BORDER_BOX = [(0.1, 2.5), (4,0.1), (4,0.1), (0.1, 2.5), (0.125, 0.5),(0.125, 0.5),(0.125, 0.5),(0.125, 0.5),
              (0.5,0.125), (0.5, 0.125),(0.5, 0.125)]  # Half of the weight and height

class ICRAMap:
    def __init__(self, world):
        self.world = world
        self.borders = [world.CreateStaticBody(
            position=p,
            shapes=polygonShape(box=b),
            ) for p, b in zip(BORDER_POS, BORDER_BOX)]
        for i in range(len(self.borders)):
            self.borders[i].color = COLOR_WHITE
            self.borders[i].userData = "wall"
        self.drawlist = self.borders

    def step(self, dt):
        pass

    def draw(self, viewer, draw_particles=True):
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                #print("ICRA Map")
                #print(path)
                viewer.draw_polygon(path, color=obj.color)

    def destroy(self):
        for border in self.borders:
            self.world.DestroyBody(border)

