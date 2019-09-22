'''
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
import random
from collections import namedtuple
from itertools import count

import numpy as np
import torch

from Agent.ActorCriticAgent import ActorCriticAgent
from Agent.HandAgent import HandAgent
from ICRAField import ICRAField
from SupportAlgorithm.NaiveMove import NaiveMove
from utils import *

move = NaiveMove()

TARGET_UPDATE = 10
ID_R1 = 0
ID_B1 = 1

seed = 233
torch.random.manual_seed(seed)
torch.cuda.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

env = ICRAField()
env.seed(seed)
agent = ActorCriticAgent()
agent2 = ActorCriticAgent()
agent.load_model()
agent2.load_model()
device = agent.device
episode_durations = []

num_episodes = 50
for i_episode in range(num_episodes):
    print("Epoch: {}".format(i_episode))
    # Initialize the environment and state
    action = Action()
    pos = env.reset()
    state, reward, done, info = env.step(action)
    for t in range(7*60*30):
        # Other agent
        env.set_robot_action(ID_B1,
                             agent2.select_action(state[ID_B1], mode="max_probability"))

        # Select and perform an action
        action = agent.select_action(state[ID_R1], "max_probability")
        # Step
        next_state, reward, done, info = env.step(action)

        state = next_state

        env.render()

        if done:
            episode_durations.append(t + 1)
            break

env.close()
