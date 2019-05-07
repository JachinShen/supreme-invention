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
from SupportAlgorithm.DataStructure import Action, RobotState

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
        env.setRobotAction("robot_1", 
            agent2.select_final_action(state["robot_1"], mode="max_probability"))

        # Select and perform an action
        action = Action()
        state_map = agent.perprocess_state(state["robot_0"])
        a_m, a_t = agent.select_action(state_map, "max_probability")
        if a_m == 0: # left
            action.v_n = -1.0
        elif a_m == 1: # ahead
            action.v_t = +1.0
        elif a_m == 2: # right
            action.v_n = +1.0

        if a_t == 0: # left
            action.omega = +1.0
        elif a_t == 1: # stay
            pass
        elif a_t == 2: # right
            action.omega = -1.0

        if state["robot_0"].detect:
            action.shoot = +1.0
        else:
            action.shoot = 0.0

        # Step
        next_state, reward, done, info = env.step(action)
        next_state_map = agent.perprocess_state(next_state["robot_0"])

        # Store the transition in memory
        agent.push(state_map, next_state_map, [a_m, a_t], [reward])
        state = next_state
        state_map = next_state_map

        env.render()

        if done:
            episode_durations.append(t + 1)
            break

env.close()
