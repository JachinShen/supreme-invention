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

SIZE = 0.001
ROBOT_SIZE = 400 // 2
WHEEL_R  = 50
WHEEL_W  = 30
ROBOT_WIDTH = ROBOT_SIZE - WHEEL_W*2
ROBOT_LENGTH = ROBOT_SIZE
HULL_POLY1 =[
    (-ROBOT_WIDTH,+ROBOT_LENGTH), (+ROBOT_WIDTH,+ROBOT_LENGTH),
    (+ROBOT_WIDTH,-ROBOT_LENGTH), (-ROBOT_WIDTH,-ROBOT_LENGTH)
    ]
WHEEL_POS_X = ROBOT_WIDTH + WHEEL_W
WHEEL_POS_Y = ROBOT_LENGTH - WHEEL_R
WHEELPOS = [
    (-WHEEL_POS_X,+WHEEL_POS_Y), (+WHEEL_POS_X,+WHEEL_POS_Y),
    (-WHEEL_POS_X,-WHEEL_POS_Y), (+WHEEL_POS_X,-WHEEL_POS_Y)
    ]
WHEEL_COLOR = (0.0,0.0,0.0)
WHEEL_WHITE = (0.3,0.3,0.3)
MUD_COLOR   = (0.4,0.4,0.0)

BULLET_BOX = (20*SIZE, 10*SIZE)
GUN_BOX = (20*SIZE, 130*SIZE)

class Robot:
    def __init__(self, world, init_angle, init_x, init_y, userData, robot_id, group='red'):
        self.world = world
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            angle = 0,
            fixtures = [
                fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY1 ]),
                    density=1.0, restitution=1),
                ]
            )
        self.hull.color = (0.8,0.0,0.0)
        self.hull.userData = userData
        self.wheels = []
        WHEEL_POLY = [
            (-WHEEL_W,+WHEEL_R), (+WHEEL_W,+WHEEL_R),
            (+WHEEL_W,-WHEEL_R), (-WHEEL_W,-WHEEL_R)
            ]
        for wx,wy in WHEELPOS[:]:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position = (init_x+wx*SIZE, init_y+wy*SIZE),
                angle = 0,
                fixtures = fixtureDef(
                    shape=polygonShape(vertices=[ (x*front_k*SIZE,y*front_k*SIZE) for x,y in WHEEL_POLY ]),
                    density=0.1,
                    restitution=1)
                    )
            print(wx, wy)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx*SIZE,wy*SIZE),
                localAnchorB=(0,0),
                enableMotor=False,
                enableLimit=True,
                #maxMotorTorque=180*900*SIZE*SIZE,
                #motorSpeed = 0,
                lowerAngle = -0.0,
                upperAngle = +0.0,
                )
            w.joint = self.world.CreateJoint(rjd)
            w.color = WHEEL_COLOR
            w.userData = userData
            self.wheels.append(w)

        self.gun = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            angle = 0,
            fixtures = [
                fixtureDef(
                    shape = polygonShape(box=GUN_BOX), 
                    maskBits=0x00,
                    density=1.0)
            ]
        )
        self.gun_joint = self.world.CreateJoint(revoluteJointDef(
            bodyA=self.hull,
            bodyB=self.gun,
            localAnchorA=(0,0),
            localAnchorB=(0,-GUN_BOX[1]),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=180*900*SIZE*SIZE,
            motorSpeed = 0.0,
            lowerAngle = -math.pi/2,
            upperAngle = +math.pi/2,
        ))
        self.gun.color = (0.1, 0.1, 0.1)
        self.hull.angle = init_angle*1.04
        self.gun.angle = init_angle
        self.drawlist =  self.wheels + [self.hull, self.gun]
        self.particles = []
        self.group = group
        self.robot_id = robot_id
        self.health = 1000.0
        self.buffLeftTime = 0
        self.command = {"ahead": 0, "rotate": 0, "transverse": 0}

    def getAnglePos(self):
        return self.gun.angle, self.gun.position

    def rotateCloudTerrance(self, angular_vel):
        self.gun_joint.motorSpeed = angular_vel

    def moveAheadBack(self, gas):
        self.command["ahead"] = gas
    
    def moveTransverse(self, transverse):
        self.command["transverse"] = transverse

    def loseHealth(self, damage):
        self.health -= damage

    def turnLeftRight(self, r):
        self.command["rotate"] = r

    def step(self, dt):
        forw = self.hull.GetWorldVector( (0,1) )  # forward
        side = self.hull.GetWorldVector( (1,0) )
        v = self.hull.linearVelocity
        vf = forw[0]*v[0] + forw[1]*v[1]  # forward speed???
        vs = side[0]*v[0] + side[1]*v[1]  # side speed
        f_force = -vf + self.command["ahead"]
        p_force = -vs + self.command["transverse"]

        f_force *= 205000*SIZE*SIZE  # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
        p_force *= 205000*SIZE*SIZE
        self.hull.ApplyForceToCenter( (
            (p_force)*side[0] + f_force*forw[0],
                                (p_force)*side[1] + f_force*forw[1]), True )
        
        torque = - self.hull.angularVelocity*0.001 + self.command["rotate"] * 0.005
        self.hull.ApplyAngularImpulse(torque, True)

    def draw(self, viewer, draw_particles=True):
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                #print(path)
                viewer.draw_polygon(path, color=obj.color)

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []
        if self.gun:
            self.world.DestroyBody(self.gun)
        self.gun = None

