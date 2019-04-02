import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 4, 2), # 5x8 -> 4x7
            nn.LeakyReLU(),
            nn.Conv2d(4, 16, 2), # 3x6
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 2), # 3x6
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 2), # 3x6
            nn.LeakyReLU(),
        )
        self.dconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(2, 2)),  # 3x6 -> 5x8
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(2, 2)),  # 5x8 -> 9x16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=(2, 2)),  # 17x28
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=(2, 2)),  # 17x28
            nn.LeakyReLU(),
        )

    def forward(self, s):
        #window_map = s[:,0:1,:,:]
        #enemy_map = s[:,1:2,:,:]
        #last_value_map = s[:,2:3,:,:]
        feature_map = self.conv(s[:,0:2,:])

        value_map = self.dconv(feature_map)

        batch, channel, w, h = value_map.shape
        value_map = value_map.reshape([batch, -1])
        value_map = F.softmax(value_map, dim=1)
        value_map = value_map.reshape([batch, channel, w, h])

        return value_map

if __name__=="__main__":
    x = torch.rand([1,3,20,32])
    model = Actor()
    y = model(x)
    print(y)