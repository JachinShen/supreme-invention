import random
import torch
import numpy as np
from collections import namedtuple
from itertools import count

from ICRAField import ICRAField
from Agent.DQNAgent import DQNAgent
from Agent.HandAgent import HandAgent
from SupportAlgorithm.NaiveMove import NaiveMove

move = NaiveMove()

TARGET_UPDATE = 100

seed = 233
torch.random.manual_seed(seed)
torch.cuda.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

agent = DQNAgent()
#agent.load()
agent.load_memory()

for epoch in range(2000):
    print("Epoch: [{}/{}]".format(epoch, 2000))
    agent.optimize_model()
    if epoch % TARGET_UPDATE == 0:
        agent.update_target_net()
        agent.save()
