import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), # 160x100 -> 158x98
            nn.ReLU(),
            nn.Conv2d(32, 64, 6, 3), # 77x47
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*12, 128),
        )
        self.head_x = nn.Sequential(
            nn.Linear(128, 32), # 8x4
            nn.Softmax(dim=1),
        )
        self.head_y = nn.Sequential(
            nn.Linear(128, 20), # 5x4
            nn.Softmax(dim=1),
        )
        '''
        self.fc_std = nn.Sequential(
            nn.Linear(64*5*1, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )
        self.dconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=3),
            nn.LeakyReLU(),
        )
        '''

    def forward(self, s):
        feature_map = self.conv(s[:,:])
        batch, channel, h, w = feature_map.shape
        feature_map = feature_map.reshape([batch, -1])
        head = self.fc(feature_map)
        x, y = self.head_x(head), self.head_y(head)
        return x, y

if __name__=="__main__":
    x = torch.rand([1,3,100,160])
    model = Actor()
    y = model(x)
    print(y)