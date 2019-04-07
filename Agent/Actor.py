import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), # 160x100 -> 158x98
            nn.ReLU(),
            nn.Conv2d(32, 64, 6, 3), # 77x47
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*12, 128),
            nn.ReLU(),
        )
        self.head_x = nn.Sequential(
            nn.Linear(128, 32), # 8x4
            nn.Softmax(dim=1),
        )
        self.head_y = nn.Sequential(
            nn.Linear(128, 20), # 5x4
            nn.Softmax(dim=1),
        )
        self.head_v = nn.Sequential(
            nn.Linear(128, 1),
        )

    def forward(self, s):
        feature_map = self.conv(s[:,:])
        batch, channel, h, w = feature_map.shape
        feature_map = feature_map.reshape([batch, -1])
        head = self.fc(feature_map)
        x, y, v = self.head_x(head), self.head_y(head), self.head_v(head)
        return x, y, v

if __name__=="__main__":
    x = torch.rand([1,3,100,160])
    model = ActorCritic()
    y = model(x)
    print(y)