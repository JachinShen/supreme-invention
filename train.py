'''
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
import random
import torch
import numpy as np
from collections import namedtuple
from itertools import count

from ICRAField import ICRAField
from DQNAgent import DQNAgent

TARGET_UPDATE = 10

seed = 233
torch.random.manual_seed(seed)
torch.cuda.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

env = ICRAField()
agent = DQNAgent()
episode_durations = []

num_episodes = 5000
for i_episode in range(num_episodes):
    print("Epoch: [{}/{}]".format(i_episode, num_episodes))
    # Initialize the environment and state
    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    env.reset()
    state, reward, done, info = env.step(action)
    for t in range(7*60*30):
        if t % (60*30) == 0:
            print("Simulation in minute: [{}/{}]".format(t//(60*30)+1, 7))
        # Select and perform an action
        if state[5] > 0:
            action[4] = +1.0
        else:
            action[4] = 0.0
        action = agent.select_action(state, action)

        next_state, reward, done, info = env.step(action)

        # Store the transition in memory
        agent.push(next_state, reward)
        state = next_state

        # Perform one step of the optimization (on the target network)
        agent.optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        print("Last reward: {}".format(reward))
        agent.update_target_net()
        agent.save()

print('Complete')
env.close()