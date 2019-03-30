import numpy as np
import math
import Box2D
from Box2D.b2 import (fixtureDef, polygonShape, )

SIZE = 0.001
BULLET_BOX = (40*SIZE, 10*SIZE)


class Bullet:
    def __init__(self, world):
        self.world = world
        self.bullets = {}
        self.ctr = 1

    def shoot(self, init_angle, init_pos):
        MIN_RANGE = 0.9
        angle = init_angle
        x, y = init_pos
        x += math.cos(angle) * MIN_RANGE
        y += math.sin(angle) * MIN_RANGE
        bullet = self.world.CreateDynamicBody(
            position=(x, y),
            angle=angle,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(box=BULLET_BOX),
                    categoryBits=0x02,
                    maskBits=0xFD,
                    density=1e-6)
            ]
        )
        bullet.bullet=True
        bullet.color = (0.0, 0.0, 0.0)
        bullet.userData = "bullet_{}".format(self.ctr)
        self.ctr += 1
        bullet.linearVelocity = (math.cos(angle)*5, math.sin(angle)*5)
        self.bullets[bullet.userData] = bullet

    def draw(self, viewer):
        for obj in self.bullets.values():
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                viewer.draw_polygon(path, color=obj.color)

    def destroyById(self, bullet_id):
        body = self.bullets.pop(bullet_id, "repeated")
        if body != "repeated":
            self.world.DestroyBody(body)

    def destroy(self):
        for bullet in self.bullets.values():
            self.world.DestroyBody(bullet)
        self.bullets = {}
