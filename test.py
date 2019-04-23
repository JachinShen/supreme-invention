'''
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
import random
from collections import namedtuple
from itertools import count

import numpy as np
import torch

from Agent.ActorCriticAgent import ActorCriticAgent
from Agent.MCTSAgent import MCTSAgent
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
#agent = ActorCriticAgent()
agent = MCTSAgent()
agent2 = HandAgent()
#agent.load_model()
#device = agent.device
episode_durations = []

num_episodes = 50
for i_episode in range(num_episodes):
    print("Epoch: {}".format(i_episode))
    # Initialize the environment and state
    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pos = env.reset()
    agent.reset()
    agent2.reset(pos)
    state, reward, done, info = env.step(action)
    goal = None
    for t in range(7*60*30):
        # Other agent
        env.setRobotAction("robot_1", agent2.select_action(
            env.getStateArray("robot_1")))
        # Select and perform an action
        state = env.state_dict
        pos = env.state_dict["robot_0"]["pos"]
        vel = env.state_dict["robot_0"]["velocity"]
        angle = env.state_dict["robot_0"]["angle"]
        if goal is None or (goal[0]-pos[0])**2 + (goal[1]-pos[1])**2 < 0.001:
            #print("Achieve goal x:{}, y:{}".format(pos[0], pos[1]))
        #if t % 30 == 0:
            goal = agent.select_action(state)
        v, omega = move.moveTo(pos, vel, angle, goal)
        action[0] = v[0]
        action[1] = omega * 3
        action[2] = v[1]
        if env.state_dict["robot_0"]["robot_1"][0] > 0:
            action[4] = +1.0
        else:
            action[4] = 0.0

        # Step
        next_state, reward, done, info = env.step(action)
        next_state = env.state_dict

        state = next_state

        env.render()

        if done:
            episode_durations.append(t + 1)
            break

env.close()
