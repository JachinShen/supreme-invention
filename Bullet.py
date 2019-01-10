import numpy as np
import math
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, shape)

SIZE = 0.001

BULLET_BOX = (4*SIZE, 1*SIZE)

class Bullet:
    def __init__(self, world, init_angle, init_x, init_y):
        self.world = world
        self.bullet = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            angle = init_angle,
            fixtures = [
                fixtureDef(
                    shape = polygonShape(box=BULLET_BOX), 
                    density=1.0)
            ]
        )
        self.bullet.color = (0.0, 0.0, 0.0)
        self.drawlist = [self.bullet]

    def draw(self, viewer):
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                #print(path)
                viewer.draw_polygon(path, color=obj.color)

    def destroy(self):
        self.world.DestroyBode(self.bullet)
    