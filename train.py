'''
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
import random
import torch
import numpy as np
from collections import namedtuple
from itertools import count

from ICRAField import ICRAField
from Agent.DQNAgent import DQNAgent
from Agent.HandAgent import HandAgent

TARGET_UPDATE = 10

seed = 233
torch.random.manual_seed(seed)
torch.cuda.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

env = ICRAField()
agent = DQNAgent()
agent2 = HandAgent()
episode_durations = []

num_episodes = 2000
for i_episode in range(num_episodes):
    print("Epoch: [{}/{}]".format(i_episode, num_episodes))
    # Initialize the environment and state
    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    env.reset()
    state, reward, done, info = env.step(action)
    for t in range(2*60*30):
        if t % (60*30) == 0:
            print("Simulation in minute: [{}:00/7:00]".format(t//(60*30)))
        # Other agent
        env.setRobotAction("robot_1", agent2.select_action(
            env.getStateArray("robot_1")))
        # Select and perform an action
        action = agent.select_action(state)

        # Step
        next_state, reward, done, info = env.step(action)

        # Store the transition in memory
        agent.push(next_state, reward)
        state = next_state

        # Perform one step of the optimization (on the target network)
        agent.optimize_model()
        if done:
            print("Simulation end in: {}:{:02d}, reward: {}".format(
                t//(60*30), t % (60*30)//30, reward))
            episode_durations.append(t + 1)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        agent.update_target_net()
        agent.save()

print('Complete')
env.close()
