import math
import random
import time
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import multivariate_normal
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from Agent.Actor import ActorCritic
from util.Grid import MARGIN, Map

BATCH_SIZE = 256

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        #self.extractor = nn.Sequential(
            #nn.Conv2d(1, 32, 8, 4),
            #nn.ReLU(),
            #nn.Conv2d(32, 64, 4, 2),
            #nn.ReLU(),
        #)
        self.extractor = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64*4*8),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        feature = self.extractor(x)
        #batch, channel, h, w = feature.shape
        #feature = feature.reshape([batch, -1])
        y = self.fc(feature)
        return y

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ob = torch.from_numpy(np.load("ob.npy")).unsqueeze(0).unsqueeze(1).double().to(device)
    model = Extractor().double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(100000):
        print("Epoch: {}".format(epoch))
        #pos = torch.zeros(1,1,50,80).double().to(device)
        index = [random.randint(0,79), random.randint(0,49)]
        #pos[0,0,index[1],index[0]] = 1
        #x = torch.cat([ob, pos], dim=1)
        x = torch.tensor([index]).double().to(device)
        y_real = ob[0,0,index[1],index[0]]
        print(y_real)
        y = model(x)
        loss = nn.MSELoss()(y, y_real)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Loss: {}".format(loss.item()))