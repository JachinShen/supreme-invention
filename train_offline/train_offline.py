import random
import sys
from collections import namedtuple
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(".")
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
agent.load_model()
agent.load_memory("resources/replay-5.memory")
agent.optimize_offline(50)
agent.save_model()

'''
losses = []
NUM_EPOCH = 1000
for epoch in range(1, NUM_EPOCH):
    print("Epoch: [{}/{}]".format(epoch, NUM_EPOCH))
    agent.optimize_model()
    loss = agent.test_model()
    losses.append(loss)
    if epoch % TARGET_UPDATE == 0:
        print("Loss: {}".format(loss))
        agent.save()
'''

print("Complete!")
'''
plt.figure(figsize=(15, 9))
plt.plot(losses)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.savefig("loss.pdf")
'''
