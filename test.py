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

move = NaiveMove()

TARGET_UPDATE = 10

seed = 233
torch.random.manual_seed(seed)
torch.cuda.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

env = ICRAField()
agent = ActorCriticAgent()
agent2 = HandAgent()
agent.load_model()
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
    for t in range(7*60*30):
        # Other agent
        env.setRobotAction("robot_1", agent2.select_action(
            env.getStateArray("robot_1")))
        # Select and perform an action
        state = env.state_dict["robot_0"]["detect"]
        state_map = agent.perprocess_state(state)
        a = agent.select_action(state_map, "max_probability")
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if a == 0:
            action[0] = +1.0
        elif a == 1:
            action[0] = -1.0
        elif a == 2:
            action[1] = +1.0
        elif a == 3:
            action[1] = -1.0
        elif a == 4:
            action[2] = +1.0
        elif a == 5:
            action[2] = -1.0
        if env.state_dict["robot_0"]["robot_1"][0] > 0:
            action[4] = +1.0
        else:
            action[4] = 0.0

        # Step
        next_state, reward, done, info = env.step(action)
        next_state = env.state_dict["robot_0"]["detect"]
        next_state_map = agent.perprocess_state(next_state)

        # Store the transition in memory
        agent.push(state, next_state, [a], [reward])
        state = next_state
        state_map = next_state_map

        env.render()

        if done:
            episode_durations.append(t + 1)
            break

env.close()
