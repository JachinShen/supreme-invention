'''
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
import random
from collections import namedtuple
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import torch

from Agent.DQNAgent import DQNAgent
from Agent.ActorCriticAgent import ActorCriticAgent
from Agent.HandAgent import HandAgent
from ICRAField import ICRAField
from SupportAlgorithm.NaiveMove import NaiveMove

move = NaiveMove()

TARGET_UPDATE = 500

seed = 233
torch.random.manual_seed(seed)
torch.cuda.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

env = ICRAField()
agent = ActorCriticAgent()
#agent.load()
agent2 = HandAgent()
episode_durations = []

num_episodes = 1001
for i_episode in range(1, num_episodes):
    print("Epoch: [{}/{}]".format(i_episode, num_episodes))
    # Initialize the environment and state
    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pos = env.reset()
    agent2.reset(pos)
    state, reward, done, info = env.step(action)
    state_map = agent.perprocess_state(state)
    for t in range(2*60*30):
        if t % (60*30) == 0:
            print("Simulation in minute: [{}:00/7:00]".format(t//(60*30)))
        # Other agent
        env.setRobotAction("robot_1", agent2.select_action(
            env.getStateArray("robot_1")))
        # Select and perform an action
        goal = agent.select_action(state_map)
        pos = (state[0], state[1])
        vel = (state[2], state[3])
        angle = state[4]
        v, omega = move.moveTo(pos, vel, angle, goal)
        action[0] = v[0]
        action[1] = omega
        action[2] = v[1]
        if state[-1] > 0 and state[-3] > 0:
            action[4] = +1.0
        else:
            action[4] = 0.0

        # Step
        next_state, reward, done, info = env.step(action)
        next_state_map = agent.perprocess_state(next_state)

        # Store the transition in memory
        agent.push(next_state_map, reward)
        state = next_state
        state_map = next_state_map

        if done:
            break

    print("Simulation end in: {}:{:02d}, reward: {}".format(
        t//(60*30), t % (60*30)//30, env.reward))
    episode_durations.append(t + 1)

    if i_episode % TARGET_UPDATE == 0:
         agent.save_memory("replay-{}.memory".format(i_episode))

print('Complete')
env.close()
