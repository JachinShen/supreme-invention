import random
from collections import namedtuple
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import torch

from Agent.DQNAgent import DQNAgent
from Agent.HandAgent import HandAgent
from Agent.ActorCriticAgent import ActorCriticAgent
from ICRAField import ICRAField
from SupportAlgorithm.NaiveMove import NaiveMove

move = NaiveMove()

TARGET_UPDATE = 100

seed = 233
torch.random.manual_seed(seed)
torch.cuda.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

agent = ActorCriticAgent()
# agent.load()
agent.load_memory("replay.memory")

losses = []
for epoch in range(800):
    print("Epoch: [{}/{}]".format(epoch, 800))
    agent.optimize_model()
    loss = agent.test_model()
    losses.append(loss)
    if epoch % TARGET_UPDATE == 0:
        print("Loss: {}".format(loss))
        agent.save()

plt.figure(figsize=(15, 9))
plt.plot(losses)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.savefig("loss.pdf")
