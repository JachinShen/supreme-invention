from Referee.SupplyArea import SUPPLYAREABOX_BLUE
from Referee.SupplyArea import SUPPLYAREABOX_RED
import numpy as np
import math
import Box2D
from Box2D.b2 import (fixtureDef, polygonShape, revoluteJointDef, )

# ICRA 2019 Robot Simulation
#
# Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
# Modified by SJTU Jiaolong.

SIZE = 0.001
ROBOT_SIZE = 400 // 2
WHEEL_R = 50
WHEEL_W = 30
ROBOT_WIDTH = ROBOT_SIZE - WHEEL_W*2
ROBOT_LENGTH = ROBOT_SIZE
HULL_POLY = [
    (-ROBOT_LENGTH, +ROBOT_WIDTH), (+ROBOT_LENGTH, +ROBOT_WIDTH),
    (+ROBOT_LENGTH, -ROBOT_WIDTH), (-ROBOT_LENGTH, -ROBOT_WIDTH)
]
WHEEL_POS_X = ROBOT_LENGTH - WHEEL_R
WHEEL_POS_Y = ROBOT_WIDTH + WHEEL_W
WHEEL_POS = [
    (-WHEEL_POS_X, +WHEEL_POS_Y), (+WHEEL_POS_X, +WHEEL_POS_Y),
    (-WHEEL_POS_X, -WHEEL_POS_Y), (+WHEEL_POS_X, -WHEEL_POS_Y)
]
WHEEL_POLY = [
    (-WHEEL_R, +WHEEL_W), (+WHEEL_R, +WHEEL_W),
    (+WHEEL_R, -WHEEL_W), (-WHEEL_R, -WHEEL_W)
]
WHEEL_COLOR = (0.0, 0.0, 0.0)

GUN_POLY = [
    (-0, +20), (+260, +20),
    (+260, -20), (-0, -20)
]

BULLETS_ADDED_ONE_TIME = 50

SUPPLY_AREAS = {
    'red': SUPPLYAREABOX_RED,  # (x, y, w, h)
    'blue': SUPPLYAREABOX_BLUE
}

class Robot:
    def __init__(self, world, init_angle, init_x, init_y, userData, robot_id, group, color):
        self.world = world
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=0,
            fixtures=[
                fixtureDef(shape=polygonShape(vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY]),
                           density=1.0, restitution=1, userData=userData, friction=1),
            ]
        )
        # self.hull.color = (0.8,0.0,0.0)
        self.hull.color = color
        self.hull.userData = userData
        self.wheels = []
        for wx, wy in WHEEL_POS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position=(init_x+wx*SIZE, init_y+wy*SIZE),
                angle=0,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[(x*front_k*SIZE, y*front_k*SIZE) for x, y in WHEEL_POLY]),
                    density=1e-6, restitution=1, userData=userData + "_wheel", friction=1)
            )
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx*SIZE, wy*SIZE),
                localAnchorB=(0, 0),
                enableMotor=False,
                enableLimit=True,
                lowerAngle=-0.0,
                upperAngle=+0.0,
            )
            w.joint = self.world.CreateJoint(rjd)
            w.color = WHEEL_COLOR
            w.userData = userData
            self.wheels.append(w)

        self.gun = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=0,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x*SIZE, y*SIZE) for x, y in GUN_POLY]),
                    categoryBits=0x00, density=1e-6, userData=userData)
            ]
        )
        self.gun_joint = self.world.CreateJoint(revoluteJointDef(
            bodyA=self.hull,
            bodyB=self.gun,
            localAnchorA=(0, 0),
            localAnchorB=(0, 0),
            enableMotor=True,
            enableLimit=False,
            maxMotorTorque=180*900*SIZE*SIZE,
            motorSpeed=0.0,
            #lowerAngle = -math.pi,
            #upperAngle = +math.pi,
        ))
        self.gun.color = (0.1, 0.1, 0.1)

        self.hull.angle = init_angle
        self.gun.angle = init_angle
        self.drawlist = self.wheels + [self.hull, self.gun]
        self.group = group
        self.robot_id = robot_id
        self.health = 1000.0
        self.buffLeftTime = 0
        self.command = {"ahead": 0, "rotate": 0, "transverse": 0}

        self.bullets_num = 40
        self.opportuniy_to_add_bullets = 2

    def refreshReloadOppotunity(self):
        self.opportuniy_to_add_bullets = 2

    def addBullets(self):
        if(self.opportuniy_to_add_bullets <= 0):
            return
        self.opportuniy_to_add_bullets -= 1
        if(self.isInSupplyArea()):
            # if(SupplyAreas.isInSupplyArea(self)):
            self.bullets_num += BULLETS_ADDED_ONE_TIME

    def isInSupplyArea(self):
        # TODO(zhouyiyuan): define the supply area and implement this function
        # return True
        if(self.group not in SUPPLY_AREAS.keys()):
            return False
        supply_area = SUPPLY_AREAS[self.group]
        x_robot, y_robot = self.hull.position.x, self.hull.position.y
        bx, by, w, h = supply_area
        if (x_robot >= bx and x_robot <= bx + w and y_robot >= by and y_robot <= by + h):
            return True
        else:
            return False

    def getGunAnglePos(self):
        return self.gun.angle, self.gun.position

    def getAnglePos(self):
        return self.hull.angle, self.hull.position

    def getPos(self):
        return self.hull.position

    def getVelocity(self):
        return self.hull.linearVelocity

    def getAngle(self):
        return self.hull.angle

    def getWorldselfVector(self):
        return self.hull.GetWorldVector

    def rotateCloudTerrance(self, angular_vel):
        self.gun_joint.motorSpeed = angular_vel

    def setCloudTerrance(self, angle):
        self.gun.angle = angle

    def moveAheadBack(self, gas):
        self.command["ahead"] = gas

    def moveTransverse(self, transverse):
        self.command["transverse"] = transverse

    def loseHealth(self, damage):
        self.health -= damage

    def turnLeftRight(self, r):
        self.command["rotate"] = r

    def step(self, dt):
        forw = self.hull.GetWorldVector((1, 0))  # forward
        side = self.hull.GetWorldVector((0, -1))
        v = self.hull.linearVelocity
        vf = forw[0]*v[0] + forw[1]*v[1]  # forward speed???
        vs = side[0]*v[0] + side[1]*v[1]  # side speed
        #f_a = (-vf + self.command["ahead"]) * 5
        #p_a = (-vs + self.command["transverse"]) * 5

        #f_force = self.hull.mass * f_a
        #p_force = self.hull.mass * p_a
        f_force = self.command["ahead"]
        p_force = self.command["transverse"]

        #self.hull.ApplyForceToCenter((
            #(p_force)*side[0] + f_force*forw[0],
            #(p_force)*side[1] + f_force*forw[1]), True)
        self.hull.linearVelocity = (
            (p_force)*side[0] + f_force*forw[0],
            (p_force)*side[1] + f_force*forw[1])

        #omega = - self.hull.angularVelocity * \
            #0.5 + self.command["rotate"] * 2
        #torque = self.hull.mass * omega
        #self.hull.ApplyTorque(torque, True)
        self.hull.angularVelocity = self.command["rotate"]

    def draw(self, viewer):
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                # print(path)
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
