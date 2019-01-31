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
    (-40,+110), (+40,+110),
    (+40,-110), (-40,-110)
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
    (-60,-120), (+60,-120),
    (+60,-90),  (-60,-90)
    ]
WHEEL_COLOR = (0.0,0.0,0.0)
WHEEL_WHITE = (0.3,0.3,0.3)
MUD_COLOR   = (0.4,0.4,0.0)

BULLET_BOX = (20*SIZE, 10*SIZE)
GUN_BOX = (50*SIZE, 20*SIZE)

class Robot:
    def __init__(self, world, init_angle, init_x, init_y,userData, robot_id, group='red'):
        self.world = world
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            angle = init_angle,
            fixtures = [
                fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY1 ]),
                    density=10.0, restitution=1),
                #fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY2 ]),
                    #density=10.0, restitution=1),
                #fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY3 ]),
                    #density=10.0, restitution=1),
                #fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY4 ]),
                    #density=10.0, restitution=1),
                ]
            )
        self.hull.color = (0.8,0.0,0.0)
        self.hull.userData = userData
        self.wheels = []
        WHEEL_POLY = [
            (-WHEEL_W,+WHEEL_R), (+WHEEL_W,+WHEEL_R),
            (+WHEEL_W,-WHEEL_R), (-WHEEL_W,-WHEEL_R)
            ]
        for wx,wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position = (init_x+wx*SIZE, init_y+wy*SIZE),
                angle = init_angle,
                fixtures = fixtureDef(
                    shape=polygonShape(vertices=[ (x*front_k*SIZE,y*front_k*SIZE) for x,y in WHEEL_POLY ]),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0)
                    )
            w.wheel_rad = front_k*WHEEL_R*SIZE
            w.color = WHEEL_COLOR
            w.gas   = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.reverse = 0.0
            w.rotation=0.0
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx*SIZE,wy*SIZE),
                localAnchorB=(0,0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180*900*SIZE*SIZE,
                motorSpeed = 0,
                lowerAngle = -0.4,
                upperAngle = +0.4,
                )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)

        self.gun = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            angle = init_angle + math.pi/2,
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
            localAnchorB=(-GUN_BOX[0],0),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=180*900*SIZE*SIZE,
            motorSpeed = 0.0,
            lowerAngle = -math.pi/2,
            upperAngle = +math.pi/2,
        ))
        self.gun.color = (0.1, 0.1, 0.1)
        self.drawlist =  self.wheels + [self.hull, self.gun]
        self.particles = []
        self.group = group
        self.robot_id = robot_id
        self.health = 1000.0
        self.buffLeftTime = 0
        self.command = {"ahead": 0, "rotate": 0, "transverse": 0}

    def getAnglePos(self):
        return self.gun.angle, self.gun.position

    def moveHead(self, angular_vel):
        self.gun_joint.motorSpeed = angular_vel

    def reverse(self, reverse):
        'control: rear wheel drive'
        reverse = np.clip(reverse, 0, 1)
        for w in self.wheels[2:4]:
            diff = reverse - w.reverse
            if diff > 0.1: diff = 0.1  # gradually increase, but stop immediately
            w.reverse += diff

    def goAhead(self, gas):
        self.command["ahead"] = gas
    
    def moveTransverse(self, transverse):
        self.command["transverse"] = transverse

    def loseHealth(self, damage):
        self.health -= damage

    def _health(self):
        return self.health

    def steer(self, s):
        'control: steer s=-1..1, it takes time to rotate steering wheel from side to side, s is target position'
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def rotation(self, r):
        self.command["rotate"] = r
        #'control: steer s=-1..1, it takes time to rotate steering wheel from side to side, s is target position'
        #self.wheels[0].rotation = r
        #self.wheels[1].rotation = r

    def step(self, dt):
        forw = self.hull.GetWorldVector( (0,1) )  # forward
        side = self.hull.GetWorldVector( (1,0) )
        v = self.hull.linearVelocity
        vf = forw[0]*v[0] + forw[1]*v[1]  # forward speed???
        vs = side[0]*v[0] + side[1]*v[1]  # side speed
        f_force = -vf + self.command["ahead"]
        p_force = -vs + self.command["transverse"]

        r_force= 0
        # Physically correct is to always apply friction_limit until speed is equal.
        # But dt is finite, that will lead to oscillations if difference is already near zero.
        f_force *= 205000*SIZE*SIZE  # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
        p_force *= 205000*SIZE*SIZE
        r_force *= 205000*SIZE*SIZE
        self.hull.ApplyForceToCenter( (
            (r_force+p_force)*side[0] + f_force*forw[0],
                                (r_force+p_force)*side[1] + f_force*forw[1]), True )
        
        torque = - self.hull.angularVelocity*0.001 + self.command["rotate"] * 0.005
        self.hull.ApplyAngularImpulse(torque, True)

    def draw(self, viewer, draw_particles=True):
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                #print(path)
                viewer.draw_polygon(path, color=obj.color)
                if "phase" not in obj.__dict__: continue
                a1 = obj.phase
                a2 = obj.phase + 1.2  # radians
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1>0 and s2>0: continue
                if s1>0: c1 = np.sign(c1)
                if s2>0: c2 = np.sign(c2)
                white_poly = [
                    (-WHEEL_W*SIZE, +WHEEL_R*c1*SIZE), (+WHEEL_W*SIZE, +WHEEL_R*c1*SIZE),
                    (+WHEEL_W*SIZE, +WHEEL_R*c2*SIZE), (-WHEEL_W*SIZE, +WHEEL_R*c2*SIZE)
                    ]
                viewer.draw_polygon([trans*v for v in white_poly], color=WHEEL_WHITE)

    def _create_particle(self, point1, point2, grass):
        class Particle:
            pass
        p = Particle()
        p.color = WHEEL_COLOR if not grass else MUD_COLOR
        p.ttl = 1
        p.poly = [(point1[0],point1[1]), (point2[0],point2[1])]
        p.grass = grass
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []
        if self.gun:
            self.world.DestroyBody(self.gun)
        self.gun = None

