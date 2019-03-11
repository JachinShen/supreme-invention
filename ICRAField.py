import math
import random
import sys

import Box2D
import gym
import numpy as np
import pyglet
from gym import spaces
from gym.utils import EzPickle, colorize, seeding
from pyglet import gl

from Objects.Bullet import Bullet
from Objects.Robot import Robot
from Referee.BuffArea import AllBuffArea
from Referee.ICRAContactListener import ICRAContactListener
from Referee.ICRAMap import ICRAMap
from Referee.SupplyArea import SupplyAreas
from SupportAlgorithm.DetectCallback import detectCallback
from SupportAlgorithm.GlobalLocalPlanner import GlobalLocalPlanner

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
COLOR_RED = (0.8, 0.0, 0.0)
COLOR_BLUE = (0.0, 0.0, 0.8)


class ICRAField(gym.Env, EzPickle):
    metadata = {
        # 'render.modes': ['human', 'rgb_array', 'state_pixels'],
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
        self.state_dict = {}
        self.actions = {}

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

        self.robots['robot_0'] = Robot(
            self.world, np.pi/2, 0.5, 0.5,
            'robot_0', 0, 'red', COLOR_RED)
        self.robots['robot_1'] = Robot(
            self.world, np.pi / 2, 6.5, 0.5,
            'robot_1', 1, 'blue', COLOR_BLUE)

        self.map = ICRAMap(self.world)
        self.bullets = Bullet(self.world)
        self.buff_areas = AllBuffArea()
        self.supply_areas = SupplyAreas()

        self.state_dict["robot_0"] = {
            "pos": (-1, -1), "angle": -1, "health": -1, "velocity": (0, 0), "angular": 0,
            "robot_0": (-1, -1), "robot_1": (-1, -1)
        }
        self.state_dict["robot_1"] = {
            "pos": (-1, -1), "angle": -1, "health": -1, "velocity": (0, 0), "angular": 0,
            "robot_0": (-1, -1), "robot_1": (-1, -1)
        }
        self.actions["robot_0"] = None
        self.actions["robot_1"] = None

        return self.step(None)[0]

    def getStateArray(self, robot_id):
        robot_state = self.state_dict[robot_id]
        pos = robot_state["pos"]
        velocity = robot_state["velocity"]
        angle = robot_state["angle"]
        angular = robot_state["angular"]
        health = robot_state["health"]
        robot_0 = robot_state["robot_0"]
        robot_1 = robot_state["robot_1"]
        return [pos[0], pos[1], velocity[0], velocity[1], angle, angular,
                robot_0[0], robot_0[1], robot_1[0], robot_1[1]]

    def stepCollision(self):
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

    def stepAction(self, robot_name, action):
        # gas, rotate, transverse, rotate cloud terrance, shoot
        self.robots[robot_name].moveAheadBack(action[0])
        self.robots[robot_name].turnLeftRight(action[1])
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

    def detectEnemy(self, robot_id):
        detected = {}
        for i in range(-170, 170, 5):
            angle, pos = self.robots[robot_id].getAnglePos()
            angle += math.pi/2
            angle += i/180*math.pi
            p1 = (pos[0] + 0.2*math.cos(angle), pos[1] + 0.2*math.sin(angle))
            p2 = (pos[0] + SCAN_RANGE*math.cos(angle),
                  pos[1] + SCAN_RANGE*math.sin(angle))
            self.world.RayCast(self.detect_callback, p1, p2)
            u = self.detect_callback.userData
            if u in self.robots.keys():
                if u not in detected.keys():
                    p = detected[u] = self.detect_callback.point
                    pos = self.robots[robot_id].getPos()
                    p = (p[0] - pos[0], p[1] - pos[1])
                    angle = math.atan2(p[1], p[0])
                    # Auto shoot
                    self.robots[robot_id].setCloudTerrance(angle)

        for robot_name in self.robots.keys():
            self.state_dict[robot_id][robot_name] = detected[robot_name] if robot_name in detected.keys(
            ) else (-1, -1)

    def updateRobotState(self, robot_id):
        self.state_dict[robot_id][robot_id] = self.robots[robot_id].getPos()
        self.state_dict[robot_id]["health"] = self.robots[robot_id].health
        self.state_dict[robot_id]["pos"] = self.robots[robot_id].getPos()
        self.state_dict[robot_id]["angle"] = self.robots[robot_id].getAngle()
        self.state_dict[robot_id]["velocity"] = self.robots[robot_id].getVelocity()
        self.state_dict[robot_id]["angular"] = self.robots[robot_id].hull.angularVelocity

    def setRobotAction(self, robot_id, action):
        self.actions[robot_id] = action

    def step(self, action):
        ###### observe ######
        for robot_name in self.robots.keys():
            self.detectEnemy(robot_name)
            self.updateRobotState(robot_name)

        ###### action ######
        self.setRobotAction("robot_0", action)
        for robot_name in self.robots.keys():
            action = self.actions[robot_name]
            if action is not None:
                self.stepAction(robot_name, action)
            self.robots[robot_name].step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        ###### Referee ######
        self.stepCollision()
        self.buff_areas.detect(
            [self.robots["robot_0"], self.robots["robot_1"]], self.t)

        ###### reward ######
        step_reward = 0
        done = False
        # First step without action, called from reset()
        if self.actions["robot_0"] is not None:
            self.reward = self.robots["robot_0"].health - \
                self.robots["robot_1"].health
            self.reward -= 0.1 * self.t * FPS
            step_reward = self.reward - self.prev_reward
            if self.robots["robot_0"].health <= 0:
                done = True
                step_reward -= 10000
            if self.robots["robot_1"].health <= 0:
                done = True
                step_reward += 10000
            self.prev_reward = self.reward

        return self.getStateArray("robot_0"), step_reward, done, {}

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


if __name__ == "__main__":
    from pyglet.window import key
    # gas, rotate, transverse, rotate cloud terrance, shoot, reload
    a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == key.ESCAPE:
            restart = True
        if k == key.W:
            a[0] = +1.0
        if k == key.S:
            a[0] = -1.0
        if k == key.Q:
            a[1] = +1.0
        if k == key.E:
            a[1] = -1.0
        if k == key.D:
            a[2] = +1.0
        if k == key.A:
            a[2] = -1.0
        if k == key.Z:
            a[3] = +1.0
        if k == key.C:
            a[3] = -1.0
        if k == key.SPACE:
            a[4] = +1.0
        if k == key.R:
            a[5] = +1.0

    def key_release(k, mod):
        if k == key.ESCAPE:
            restart = True
        if k == key.W:
            a[0] = +0.0
        if k == key.S:
            a[0] = -0.0
        if k == key.Q:
            a[1] = +0.0
        if k == key.E:
            a[1] = -0.0
        if k == key.D:
            a[2] = +0.0
        if k == key.A:
            a[2] = -0.0
        if k == key.Z:
            a[3] = +0.0
        if k == key.C:
            a[3] = -0.0
        if k == key.SPACE:
            a[4] = +0.0
        if k == key.R:
            a[5] = +0.0

    env = ICRAField()
    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    move = GlobalLocalPlanner()
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        s, r, done, info = env.step(a)
        pos = (s[0], s[1])
        vel = (s[2], s[3])
        ang = s[4]
        target = (2.5, 4.5)   # origin (0.5, 0.5)
        move.setGoal(pos, target, 0.0)
        while True:
            s, r, done, info = env.step(a)
            pos = (s[0], s[1])
            vel = (s[2], s[3])
            angle = s[4]
            angular = env.state_dict["robot_0"]["angular"]
            a = move.moveTo(pos, vel, angle, angular, a)
            # a = agent.run(s, a) # Dont Shoot yet
            total_reward += r

            if steps % 200 == 0 or done:
                #     print("state: {}".format(s))
                #     print("action " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            # Faster, but you can as well call env.render() every time to play full window.
            if not record_video:
                env.render()
            if done or restart:
                break
    env.close()
