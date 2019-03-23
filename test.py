'''
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
import random
from collections import namedtuple
from itertools import count

import numpy as np
import torch

from Agent.DQNAgent import DQNAgent
from Agent.HandAgent import HandAgent
from ICRAField import ICRAField
from SupportAlgorithm.NaiveMove import NaiveMove

move = NaiveMove()

TARGET_UPDATE = 10

seed = 14
torch.random.manual_seed(seed)
torch.cuda.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

env = ICRAField()
agent = DQNAgent()
agent2 = HandAgent()
agent.load()
device = agent.device
episode_durations = []

num_episodes = 50
for i_episode in range(num_episodes):
    print("Epoch: {}".format(i_episode))
    # Initialize the environment and state
    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pos = env.reset()
    agent2.reset(pos)
    state, reward, done, info = env.step(action)
    state_obs = agent.perprocess_state(state)
    for t in range(7*60*30):
        if t % (60*30) == 0:
            print("Simulation in minute: [{}:00/7:00]".format(t//(60*30)))
        env.setRobotAction("robot_1", agent2.select_action(
            env.getStateArray("robot_1")))
        # Select and perform an action
        '''
        if state[5] > 0:
            action[4] = +1.0
        else:
            action[4] = 0.0
        '''
        #action = agent.select_action(state, True)
        #action = agent.select_action(state, state_obs, True)
        goal = agent.select_action(state, state_obs, True)
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

        next_state, reward, done, info = env.step(action)
        next_state_obs = agent.perprocess_state(next_state)

        # Move to the next state
        state = next_state
        state_obs = next_state_obs
        env.render()

        if done:
            episode_durations.append(t + 1)
            #agent2.reset()
            break

env.close()
