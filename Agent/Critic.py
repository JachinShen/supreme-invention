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
            nn.Conv2d(3, 8, 3), # 160x100 -> 158x98
            nn.MaxPool2d(2, 2), # 79x49
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 3), # 77x47
            nn.MaxPool2d(2, 2), # 39x24
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3), # 37x22
            nn.MaxPool2d(2, 2), # 19x11
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3), # 17x9
            nn.MaxPool2d(2, 2), # 9x5
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3), # 7x3
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 2), # 5x1
        )
        self.fc = nn.Sequential(
            nn.Linear(64*5, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, s):
        feature_map = self.conv(s)
        batch, channel, w, h = feature_map.shape
        feature_map = feature_map.reshape([batch, -1])
        value_eval = self.fc(feature_map)

        return value_eval

if __name__=="__main__":
    x = torch.rand([1,3,20,32])
    model = Critic()
    y = model(x)
    print(y)