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
from Referee.BuffArea import BuffArea
from Referee.ICRAContactListener import ICRAContactListener
from Referee.ICRAMap import ICRAMap
from Referee.SupplyArea import SupplyAreas
from SupportAlgorithm.DetectCallback import detectCallback
#from SupportAlgorithm.DataStructure import Action, RobotState
from utils import *

WINDOW_W = 1200
WINDOW_H = 1000

SCALE = 40.0        # Track scale
PLAYFIELD = 400/SCALE  # Game over boundary
FPS = 30
ZOOM = 2.7        # Camera zoom

SCAN_RANGE = 5  # m

ID_R1 = 0
ID_B1 = 1


def robotName2ID(robot_name):
    if robot_name == "robot_0":
        return ID_R1
    elif robot_name == "robot_1":
        return ID_B1


class ICRAField(gym.Env, EzPickle):

    __pos_safe = [
        [0.5, 0.5], [0.5, 2.0], [0.5, 3.0], [0.5, 4.5],  # 0 1 2 3
        [1.5, 0.5], [1.5, 3.0], [1.5, 4.5],             # 4 5 6
        [2.75, 0.5], [2.75, 2.0], [2.75, 3.0], [2.75, 4.5],  # 7 8 9 10
        [4.0, 1.75], [4.0, 3.25],                         # 11 12
        [5.25, 0.5], [5.25, 2.0], [5.25, 3.0], [5.25, 4.5],  # 13 14 15 16
        [6.5, 0.5], [6.5, 2.0], [6.5, 4.5],             # 17 18 19
        [7.5, 0.5], [7.5, 2.0], [7.5, 3.0], [7.5, 4.5]  # 20 21 22 23
    ]
    __id_pos_linked = [
        [1, 2, 3, 4], [0, 2, 3], [0, 1, 3, 5], [0, 1, 2, 6],
        [0, 7], [2, 9], [3, 10],
        [8, 9, 10, 4], [7, 9, 10, 11], [7, 8, 10, 5, 12], [7, 8, 9],
        [8, 14], [9, 15],
        [14, 15, 16, 17], [13, 15, 16, 18, 11, 11, 11, 11, 11], [
            13, 14, 16, 12, 12, 12, 12, 12], [13, 14, 15, 19],
        [13, 20], [14, 21], [16, 23],
        [21, 22, 23, 17], [20, 22, 23, 18], [20, 21, 23], [20, 21, 22, 19]
    ]

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.__contactListener_keepref = ICRAContactListener(self)
        self.__world = Box2D.b2World(
            (0, 0), contactListener=self.__contactListener_keepref)
        self.viewer = None
        self.__robots = []
        self.__robot_name = [ID_R1, ID_B1]
        self.__obstacle = None
        self.__area_buff = None
        self.__projectile = None
        self.__area_supply = None
        self.__callback_autoaim = detectCallback()

        self.reward = 0.0
        self.prev_reward = 0.0
        self.actions = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        for r in self.__robots:
            if r:
                r.destroy()
            r = None
        if self.__obstacle:
            self.__obstacle.destroy()
        self.__obstacle = None
        if self.__projectile:
            self.__projectile.destroy()
        self.__projectile = None

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.t = 0.0

        random_index = random.randint(0, 23)
        #random_index = 5
        init_pos_0 = self.__pos_safe[random_index]
        init_pos_1 = self.__pos_safe[random.choice(
            self.__id_pos_linked[random_index])]

        self.__R1 = Robot(self.__world, 0, init_pos_0, ID_R1)
        self.__B1 = Robot(self.__world, 0, init_pos_1, ID_B1)
        self.__robots = [self.__R1, self.__B1]

        self.__obstacle = ICRAMap(self.__world)
        self.__projectile = Bullet(self.__world)
        self.__area_buff = BuffArea()
        self.__area_supply = SupplyAreas()

        self.state = [RobotState(init_pos_0), RobotState(init_pos_1)]
        self.actions = [Action(), Action()]

        self.reward = 0

        return init_pos_1
        # return self.step(None)[0]

    def __step_contact(self):
        contact_bullet_robot = self.__contactListener_keepref.collision_bullet_robot
        contact_bullet_wall = self.__contactListener_keepref.collision_bullet_wall
        contact_robot_wall = self.__contactListener_keepref.collision_robot_wall
        contact_robot_robot = self.__contactListener_keepref.collision_robot_robot
        for bullet, robot in contact_bullet_robot:
            self.__projectile.destroyById(bullet.id)
            if(self.__robots[robot.id].buff_left_time) > 0:
                self.__robots[robot.id].lose_health(25)
            else:
                self.__robots[robot.id].lose_health(50)
        for bullet in contact_bullet_wall:
            self.__projectile.destroyById(bullet.id)
        for robot in contact_robot_wall:
            self.__robots[robot.id].lose_health(100)
        for robot in contact_robot_robot:
            self.__robots[robot.id].lose_health(10)
        self.__contactListener_keepref.clean()

    def _step_action(self, robot: Robot, action: Action):
        # gas, rotate, transverse, rotate cloud terrance, shoot
        robot.move_ahead_back(action.v_t)
        robot.turn_left_right(action.angular)
        robot.move_left_right(action.v_n)
        if int(self.t * FPS) % (60 * FPS) == 0:
            robot.refresh_supply_oppotunity()
        if action.supply > 0.99:
            action.supply = 0.0
            if robot.if_supply_available():
                robot.use_supply_oppotunity()
                if self.__area_supply.if_in_area(robot):
                    robot.supply()
        if action.shoot > 0.99 and int(self.t*FPS) % (FPS/5) == 1:
            if(robot.if_left_projectile()):
                angle, pos = robot.get_gun_angle_pos()
                robot.shoot()
                self.__projectile.shoot(angle, pos)

    def _autoaim(self, robot: Robot, state: RobotState):
        #detected = {}
        scan_distance, scan_type = [], []
        state.detect = False
        for i in range(-135, 135, 2):
            angle, pos = robot.get_angle_pos()
            angle += i/180*math.pi
            p1 = (pos[0] + 0.3*math.cos(angle), pos[1] + 0.3*math.sin(angle))
            p2 = (pos[0] + SCAN_RANGE*math.cos(angle),
                  pos[1] + SCAN_RANGE*math.sin(angle))
            self.__world.RayCast(self.__callback_autoaim, p1, p2)
            scan_distance.append(self.__callback_autoaim.fraction)
            u = self.__callback_autoaim.userData
            if u is not None and u.type == "robot":
                scan_type.append(1)
                if not state.detect:
                    robot.set_gimbal(angle)
                    state.detect = True
            else:
                scan_type.append(0)
        state.scan = [scan_distance, scan_type]

    def _update_robot_state(self, robot: Robot, state: RobotState):
        state.pos = robot.get_pos()
        state.health = robot.get_health()
        state.angle = robot.get_angle()
        state.velocity = robot.get_velocity()
        state.angular = robot.get_angular()

    def set_robot_action(self, robot_id, action: Action):
        self.actions[robot_id] = action

    def step(self, action: Action):
        ###### observe ######
        for robot, state in zip(self.__robots, self.state):
            self._autoaim(robot, state)
            self._update_robot_state(robot, state)

        ###### action ######
        self.set_robot_action(ID_R1, action)
        for robot, action in zip(self.__robots, self.actions):
            if action is not None:
                self._step_action(robot, action)
            robot.step(1.0/FPS)
        self.__world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        ###### Referee ######
        self.__step_contact()
        for robot in self.__robots:
            self.__area_buff.detect(robot, self.t)

        ###### reward ######
        step_reward = 0
        done = False
        # First step without action, called from reset()
        if self.actions[ID_R1] is not None:
            self.reward = (self.__robots[ID_R1].get_health() -
                           self.__robots[ID_B1].get_health()) / 4000.0

            #self.reward += 10 * self.t * FPS
            step_reward = self.reward - self.prev_reward
            if self.state[ID_R1].detect:
                step_reward += 1/3000

            if self.__robots[ID_R1].get_health() <= 0:
                done = True
                #step_reward -= 1
            if self.__robots[ID_B1].get_health() <= 0:
                done = True
                #step_reward += 1
            #self.reward += step_reward
            self.prev_reward = self.reward

        return self.state, step_reward, done, {}

    @staticmethod
    def get_gl_text(x, y):
        return pyglet.text.Label('0000', font_size=16, x=x, y=y,
                                 anchor_x='left', anchor_y='center',
                                 color=(255, 255, 255, 255))

    def render(self, mode='god'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.time_label = self.get_gl_text(20, WINDOW_H * 5.0 / 40.0)
            self.score_label = self.get_gl_text(520, WINDOW_H * 2.5 / 40.0)
            self.health_label = self.get_gl_text(520, WINDOW_H * 3.5 / 40.0)
            self.projectile_label = self.get_gl_text(
                520, WINDOW_H * 4.5 / 40.0)
            self.buff_left_time_label = self.get_gl_text(
                520, WINDOW_H * 5.5 / 40.0)
            self.buff_stay_time = self.get_gl_text(520, WINDOW_H * 6.5 / 40.0)
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        zoom = ZOOM*SCALE
        scroll_x = 4.0
        scroll_y = 0.0
        angle = 0
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) -
                          scroll_y*zoom*math.sin(angle)),
            WINDOW_H/4 - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)))

        self.__obstacle.draw(self.viewer)
        if mode == 'god':
            for robot in self.__robots:
                robot.draw(self.viewer)
        elif mode == "fps":
            self.__robots[ID_R1].draw(self.viewer)
            self.__robots[ID_B1].draw(self.viewer)
        self.__projectile.draw(self.viewer)

        arr = None
        win = self.viewer.window
        if mode != 'state_pixels':
            win.switch_to()
            win.dispatch_events()

        win.clear()
        t = self.transform
        gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
        t.enable()
        self._render_background()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        t.disable()
        self._render_indicators(WINDOW_W, WINDOW_H)
        win.flip()

        self.viewer.onetime_geoms = []
        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _render_background(self):
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
        self.__area_buff.render(gl)
        self.__area_supply.render(gl)

    def _render_indicators(self, W, H):
        self.time_label.text = "Time: {} s".format(int(self.t))
        self.score_label.text = "Score: %04i" % self.reward
        self.health_label.text = "health left Car0 : {} Car1: {} ".format(
            self.__robots[ID_R1].get_health(), self.__robots[ID_B1].get_health())
        self.projectile_label.text = "Car0 bullets : {}, oppotunity to add : {}  ".format(
            self.__robots[ID_R1].get_left_projectile(
            ), self.__robots[ID_R1].supply_opportunity_left
        )
        self.buff_stay_time.text = 'Buff Stay Time: Red {}s, Blue {}s'.format(
            int(self.__area_buff.get_single_buff(GROUP_RED).get_stay_time()),
            int(self.__area_buff.get_single_buff(GROUP_BLUE).get_stay_time()))
        self.buff_left_time_label.text = 'Buff Left Time: Red {}s, Blue {}s'.format(
            int(self.__robots[ID_R1].buff_left_time),
            int(self.__robots[ID_B1].buff_left_time))
        self.time_label.draw()
        self.score_label.draw()
        self.health_label.draw()
        self.projectile_label.draw()
        self.buff_stay_time.draw()
        self.buff_left_time_label.draw()


if __name__ == "__main__":
    from pyglet.window import key, mouse
    # gas, rotate, transverse, rotate cloud terrance, shoot, reload
    #a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #target = [0, 0]
    a = Action()

    def on_mouse_release(x, y, button, modifiers):
        x_low, x_high, y_low, y_high = 168, 1033, 249, 789
        width = x_high - x_low
        height = y_high - y_low
        x = (x - x_low) / width * 8.0
        y = (y - y_low) / height * 5.0
        target[0] = x
        target[1] = y

    def key_press(k, mod):
        global restart
        if k == key.ESCAPE:
            restart = True
        if k == key.W:
            a.v_t = +1.0
        if k == key.S:
            a.v_t = -1.0
        if k == key.Q:
            a.angular = +1.0
        if k == key.E:
            a.angular = -1.0
        if k == key.D:
            a.v_n = +1.0
        if k == key.A:
            a.v_n = -1.0
        if k == key.SPACE:
            a.shoot = +1.0
        if k == key.R:
            a.supply = +1.0

    def key_release(k, mod):
        if k == key.W:
            a.v_t = +0.0
        if k == key.S:
            a.v_t = -0.0
        if k == key.Q:
            a.angular = +0.0
        if k == key.E:
            a.angular = -0.0
        if k == key.D:
            a.v_n = +0.0
        if k == key.A:
            a.v_n = -0.0
        if k == key.SPACE:
            a.shoot = +0.0

    env = ICRAField()
    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    #env.viewer.window.on_mouse_release = on_mouse_release
    #move = NaiveMove()
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        s, r, done, info = env.step(a)
        while True:
            s, r, done, info = env.step(a)
            total_reward += r

            if steps % 200 == 0 or done:
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1

            # Faster, but you can as well call env.render() every time to play full window.
            if not record_video:
                env.render()
            if done or restart:
                break
    env.close()
