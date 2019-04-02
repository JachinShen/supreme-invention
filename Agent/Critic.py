import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 4, 3), # 20x32 -> 18x30
            nn.MaxPool2d(2, 2), # 9x15
            nn.LeakyReLU(),
            nn.Conv2d(4, 16, 2), # 8x14
            nn.MaxPool2d(2, 2), # 4x7
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3), # 2x5
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 2), # 1x4
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, s):
        feature_map = self.conv(s[:,0:2,:])
        batch, channel, w, h = feature_map.shape
        feature_map = feature_map.reshape([batch, -1])
        value_eval = self.fc(feature_map)

        return value_eval

if __name__=="__main__":
    x = torch.rand([1,3,20,32])
    model = Critic()
    y = model(x)
    print(y)