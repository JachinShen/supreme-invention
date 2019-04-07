'''
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
import random
import time
import sys
from collections import namedtuple
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(".")
from Agent.ActorCriticAgent import ActorCriticAgent
from Agent.HandAgent import HandAgent
from ICRAField import ICRAField
from SupportAlgorithm.NaiveMove import NaiveMove

move = NaiveMove()

TARGET_UPDATE = 5

seed = 233
torch.random.manual_seed(seed)
torch.cuda.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

env = ICRAField()
agent = ActorCriticAgent()
#agent.load_model()
agent2 = HandAgent()
episode_durations = []

a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
goal = [0, 0]

from pyglet.window import key

def on_mouse_release(x, y, button, modifiers):
    x_low, x_high, y_low, y_high = 168, 1033, 249, 789
    width = x_high - x_low
    height = y_high - y_low
    x = (x - x_low) / width * 8.0
    y = (y - y_low) / height * 5.0
    goal[0] = x
    goal[1] = y

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

env.render()
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release
env.viewer.window.on_mouse_release = on_mouse_release

num_episodes = 1001
for i_episode in range(1, num_episodes):
    print("Epoch: [{}/{}]".format(i_episode, num_episodes))
    # Initialize the environment and state
    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pos = env.reset()
    agent2.reset(pos)
    state, reward, done, info = env.step(action)
    #state_map = agent.perprocess_state(state)
    goal[:] = state[:2]
    for t in range(2*60*30):
        if t % (60*30) == 0:
            print("Simulation in minute: [{}:00/7:00]".format(t//(60*30)))
        # Other agent
        env.setRobotAction("robot_1", agent2.select_action(
            env.getStateArray("robot_1")))
        # Select and perform an action
        #goal = agent.select_action(state_map)
        pos = (state[0], state[1])
        vel = (state[2], state[3])
        angle = state[4]
        v, omega = move.moveTo(pos, vel, angle, goal)
        action[0] = v[0]
        action[1] = omega
        action[2] = v[1]
        #action = a
        if state[-1] > 0 and state[-3] > 0:
            action[4] = +1.0
        else:
            action[4] = 0.0

        # Step
        next_state, reward, done, info = env.step(action)
        #next_state_map = agent.perprocess_state(next_state)

        # Store the transition in memory
        #if state[-1] > 0 and state[-3] > 0:
        agent.push([state[:2], state[-2:]],
                [next_state[:2], next_state[-2:]], goal, [reward])
        state = next_state
        #state_map = next_state_map

        env.render("fps")
        time.sleep(1/30.0)
        if done:
            break

    agent.memory.finish_epoch()
    print("Simulation end in: {}:{:02d}, reward: {}".format(
        t//(60*30), t % (60*30)//30, env.reward))
    episode_durations.append(t + 1)

    if i_episode % TARGET_UPDATE == 0:
         agent.save_memory("resources/replay-{}.memory".format(i_episode))

print('Complete')
env.close()
