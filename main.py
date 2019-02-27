import sys
import math
import numpy as np

import Box2D

import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle

import pyglet
from pyglet import gl

from Objects.Robot import Robot
from Objects.Bullet import Bullet
from Referee.ICRAMap import ICRAMap
from Referee.BuffArea import AllBuffArea
from Referee.SupplyArea import SupplyAreas
from Referee.ICRAContactListener import ICRAContactListener
from SupportAlgorithm.DetectCallback import detectCallback
from SupportAlgorithm.MoveAction import MoveAction

STATE_W = 96   # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1200
WINDOW_H = 1000

SCALE = 40.0        # Track scale
PLAYFIELD = 400/SCALE  # Game over boundary
FPS = 30
ZOOM = 2.7        # Camera zoom
ZOOM_FOLLOW = True       # Set to False for fixed view (don't use zoom)

SCAN_RANGE = 5

class ICRAField(gym.Env, EzPickle):
    metadata = {
        #'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'render.modes': 'human',
        'video.frames_per_second': FPS
    }

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = ICRAContactListener(self)
        self.world = Box2D.b2World(
            (0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.robots = {}
        self.map = None
        self.buff_areas = None
        self.bullets = None
        self.supply_areas = None
        self.detect_callback = detectCallback()

        self.reward = 0.0
        self.prev_reward = 0.0
        # gas, rotate, transverse, rotate cloud terrance, shoot
        #self.action_space = spaces.Box(
            #np.array([-1, -1, -1, -1, 0]),
            #np.array([+1, -1, +1, +1, +1]), dtype=np.float32)
        # pos(x,y) x2, health
        #self.observation_space = spaces.Box(
            #np.array([-1, -1, -1, -1, -1]),
            #np.array([+10, +10, +10, +10, +1000]), dtype=np.float32)
        self.state = {"pos": (-1, -1), "angle": -1, "robot_1": (-1, -1), "health": -1, "velocity": (0, 0)}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        for robot_name in self.robots.keys():
            self.robots[robot_name].destroy()
        self.robots = {}
        if self.map:
            self.map.destroy()
        self.map = None
        if self.bullets:
            self.bullets.destroy()
        self.bullets = None

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.t = 0.0
        self.human_render = False

        self.robots = {}


        # for robot_name, x in zip(["robot_0", "robot_1"], [0.5, 6.5]):
        #     self.robots[robot_name] = Robot(
        #          self.world, -np.pi/2, x, 4.5, robot_name, 0, 'red')
        red_color = (0.8,0.0,0.0)
        blue_color = (0.0, 0.0, 0.8)
        self.robots['robot_0'] = Robot(self.world, -np.pi/2, 0.5, 4.5, 'robot_0', 0, 'red', red_color)
        self.robots['robot_1'] = Robot(self.world, -np.pi / 2, 6.5, 4.5, 'robot_1', 1, 'blue', blue_color)

        self.map = ICRAMap(self.world)
        self.bullets = Bullet(self.world)
        self.buff_areas = AllBuffArea()
        self.supply_areas = SupplyAreas()

        return self.step(None)[0]

    def collision_step(self):
        collision_bullet_robot = self.contactListener_keepref.collision_bullet_robot
        collision_bullet_wall = self.contactListener_keepref.collision_bullet_wall
        collision_robot_wall = self.contactListener_keepref.collision_robot_wall
        for bullet, robot in collision_bullet_robot:
            self.bullets.destroyById(bullet)
            if(self.robots[robot].buffLeftTime) > 0:
                self.robots[robot].loseHealth(25)
            else:
                self.robots[robot].loseHealth(50)
        for bullet in collision_bullet_wall:
            self.bullets.destroyById(bullet)
        for robot in collision_robot_wall:
            self.robots[robot].loseHealth(10)
        self.contactListener_keepref.clean()

    def action_step(self, robot_name, action):
        # gas, rotate, transverse, rotate cloud terrance, shoot
        self.robots[robot_name].moveAheadBack(action[0])
        self.robots[robot_name].turnLeftRight(action[1]/2)
        self.robots[robot_name].moveTransverse(action[2])
        self.robots[robot_name].rotateCloudTerrance(action[3])
        #print(int(self.t * FPS) % (60 * FPS))
        if int(self.t * FPS) % (60 * FPS) == 0:
            self.robots[robot_name].refreshReloadOppotunity()
        if action[5] > 0.99:
            self.robots[robot_name].addBullets()
            action[5] = +0.0
        if action[4] > 0.99 and int(self.t*FPS) % (FPS/5) == 1:
            if(self.robots[robot_name].bullets_num > 0):
                init_angle, init_pos = self.robots[robot_name].getGunAnglePos()
                self.bullets.shoot(init_angle, init_pos)
                self.robots[robot_name].bullets_num -= 1

    def detect_step(self):
        detected = {}
        # self.robots["robot_0"].setCloudTerrance(1)
        for i in range(-15, 15):
            angle, pos = self.robots["robot_0"].getGunAnglePos()
            angle += i/180*math.pi
            p1 = (pos[0] + 0.5*math.cos(angle), pos[1] + 0.5*math.sin(angle))
            p2 = (pos[0] + SCAN_RANGE*math.cos(angle), pos[1] + SCAN_RANGE*math.sin(angle))
            self.world.RayCast(self.detect_callback, p1, p2)
            u = self.detect_callback.userData
            if u in self.robots.keys():
                if u not in detected.keys():
                    p = detected[u] = self.detect_callback.point
                    pos = self.robots["robot_0"].getPos()
                    p = (p[0] - pos[0], p[1] - pos[1])
                    angle = math.atan2(p[1], p[0])
                    self.robots["robot_0"].setCloudTerrance(angle)


        for robot_name in self.robots.keys():
            if robot_name in detected.keys():
                self.state[robot_name] = detected[robot_name]
            else:
                self.state[robot_name] = (-1, -1)

    def step(self, action):
        self.collision_step()
        if action is not None:
            self.action_step("robot_0", action)

        self.detect_step()
        self.buff_areas.detect([self.robots["robot_0"], self.robots["robot_1"]], self.t)

        for robot_name in self.robots.keys():
            self.robots[robot_name].step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        self.state["health"] = self.robots["robot_0"].health
        self.state["pos"] = self.robots["robot_0"].getPos()
        self.state["angle"] = self.robots["robot_0"].getAngle()
        self.state["velocity"] = self.robots["robot_0"].getVelocity()

        step_reward = 0
        done = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            step_reward = self.reward - self.prev_reward
            if self.robots["robot_0"].health <= 0:
                done = True
                step_reward -= 1000
            if self.robots["robot_1"].health <= 0:
                done = True
                step_reward += 1000
            self.prev_reward = self.reward

        return self.state, step_reward, done, {}

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.time_label = pyglet.text.Label('0000', font_size=36,
                                                 x=20, y=WINDOW_H * 5.0 / 40.00, anchor_x='left', anchor_y='center',
                                                 color=(255, 255, 255, 255))
            self.score_label = pyglet.text.Label('0000', font_size=36,
                                                 x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                                                 color=(255, 255, 255, 255))
            self.health_label = pyglet.text.Label('0000', font_size=16,
                                                  x=520, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                                                  color=(255, 255, 255, 255))
            self.bullets_label = pyglet.text.Label('0000', font_size=16,
                                                   x=520, y=WINDOW_H*3.5/40.00, anchor_x='left', anchor_y='center',
                                                   color=(255, 255, 255, 255))
            self.buff_stay_time = pyglet.text.Label('0000', font_size=16,
                                                   x=520, y=WINDOW_H*4.5/40.00, anchor_x='left', anchor_y='center',
                                                   color=(255, 255, 255, 255))
            self.buff_left_time = pyglet.text.Label('0000', font_size=16,
                                                    x=520, y=WINDOW_H * 5.5 / 40.00, anchor_x='left', anchor_y='center',
                                                    color=(255, 255, 255, 255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        zoom = ZOOM*SCALE
        zoom_state = ZOOM*SCALE*STATE_W/WINDOW_W
        zoom_video = ZOOM*SCALE*VIDEO_W/WINDOW_W
        #scroll_x = self.car0.hull.position[0]
        #scroll_y = self.car0.hull.position[1]
        #angle = -self.car0.hull.angle
        scroll_x = 4.0
        scroll_y = 0.0
        angle = 0
        #vel = self.car0.hull.linearVelocity
        # if np.linalg.norm(vel) > 0.5:
        #angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) -
                          scroll_y*zoom*math.sin(angle)),
            WINDOW_H/4 - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)))
        # self.transform.set_rotation(angle)

        self.map.draw(self.viewer)
        for robot_name in self.robots.keys():
            self.robots[robot_name].draw(self.viewer)
        self.bullets.draw(self.viewer)

        arr = None
        win = self.viewer.window
        if mode != 'state_pixels':
            win.switch_to()
            win.dispatch_events()

        if mode == 'human':
            self.human_render = True
            win.clear()
            t = self.transform
            gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
            t.enable()
            self.render_background()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H)
            win.flip()

        self.viewer.onetime_geoms = []
        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_background(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD/20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k*x + k, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + k, 0)
                gl.glVertex3f(k*x + k, k*y + k, 0)
        gl.glEnd()
        self.buff_areas.render(gl)
        self.supply_areas.render(gl)

        # self.render_buff_area(self.map.buff_area)

    # def render_buff_area(self, buff_area):
    #     gl.Begin(gl.GL_QUADS)
    #     gl.glColor4f(1.0, 0.0, 0.0, 0.5)
    #     for pos, box in buff_area:
    #         pass

    def render_indicators(self, W, H):
        self.time_label.text = "Time: {} s".format(int(self.t))
        self.score_label.text = "Score: %04i" % self.reward
        self.health_label.text = "health left Car0 : {} Car1: {} ".format(
            self.robots["robot_0"].health, self.robots["robot_1"].health)
        self.bullets_label.text = "Car0 bullets : {}, oppotunity to add : {}  ".format(
            self.robots['robot_0'].bullets_num, self.robots['robot_0'].opportuniy_to_add_bullets
        )
        self.buff_stay_time.text = 'Buff Stay Time: Red {}s, Blue {}s'.format(int(self.buff_areas.buffAreas[0].maxStayTime),
                                                                    int(self.buff_areas.buffAreas[1].maxStayTime))
        self.buff_left_time.text = 'Buff Left Time: Red {}s, Blue {}s'.format(int(self.robots['robot_0'].buffLeftTime),
                                                                         int(self.robots['robot_1'].buffLeftTime))
        self.time_label.draw()
        self.score_label.draw()
        self.health_label.draw()
        self.bullets_label.draw()
        self.buff_stay_time.draw()
        self.buff_left_time.draw()

class NaiveAgent():
    def __init__(self):
        pass

    def run(self, observation, action):
        pos = observation["pos"]
        angle = observation["angle"]
        robot_1 = observation["robot_1"]
        if robot_1[0] > 0 and robot_1[1] > 0:
            action[4] = +1.0
        else:
            action[4] = +0.0
        return action


if __name__ == "__main__":
    from pyglet.window import key
    # gas, rotate, transverse, rotate cloud terrance, shoot, reload
    a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == key.ESCAPE: restart = True
        if k == key.W: a[0] = +1.0
        if k == key.S: a[0] = -1.0
        if k == key.Q: a[1] = +1.0
        if k == key.E: a[1] = -1.0
        if k == key.D: a[2] = +1.0
        if k == key.A: a[2] = -1.0
        if k == key.Z: a[3] = +1.0
        if k == key.C: a[3] = -1.0
        if k == key.SPACE: a[4] = +1.0
        if k == key.R: a[5] = +1.0

    def key_release(k, mod):
        if k == key.ESCAPE: restart = True
        if k == key.W: a[0] = +0.0
        if k == key.S: a[0] = -0.0
        if k == key.Q: a[1] = +0.0
        if k == key.E: a[1] = -0.0
        if k == key.D: a[2] = +0.0
        if k == key.A: a[2] = -0.0
        if k == key.Z: a[3] = +0.0
        if k == key.C: a[3] = -0.0
        if k == key.SPACE: a[4] = +0.0
        if k == key.R: a[5] = +0.0

    agent = NaiveAgent()

    env = ICRAField()
    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        s, r, done, info = env.step(a)
        target = Box2D.b2Vec2(6, 4.5)
        move = MoveAction(target, s)
        while True:
            s, r, done, info = env.step(a)
            a = move.MoveTo(s, a)
            # a = agent.run(s, a) # Dont Shoot yet
            total_reward += r

            # if steps % 200 == 0 or done:
            #     print("state: {}".format(s))
            #     print("action " + str(["{:+0.2f}".format(x) for x in a]))
            #     print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                #import matplotlib.pyplot as plt
                # plt.imshow(s)
                # plt.savefig("test.jpeg")
            steps += 1
            # Faster, but you can as well call env.render() every time to play full window.
            if not record_video:
                env.render()
            if done or restart:
                break
    env.close()
