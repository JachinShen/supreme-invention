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
    env.reset()
    agent2.reset()
    state, reward, done, info = env.step(action)
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
        action = agent.select_action(state, True)

        next_state, reward, done, info = env.step(action)

        # Move to the next state
        state = next_state
        env.render()

        if done:
            episode_durations.append(t + 1)
            agent2.reset()
            break

env.close()
