import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
from torch.autograd import Variable
sys.path.append(".")
from util.Grid import Map


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        #self.fc = nn.Sequential(
            #nn.Linear(16*3*6, 64),
            #nn.BatchNorm2d(64),
            #nn.LeakyReLU(),
            #nn.Linear(64, 16*3*6),
        #)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3), # 5x8 -> 4x7
            nn.MaxPool2d(2, 2),
            #nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 3), # 3x6
            nn.MaxPool2d(2, 2),
            #nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            #nn.Conv2d(16, 32, 3), # 3x6
            #nn.MaxPool2d(2, 2),
            #nn.BatchNorm2d(32),
            #nn.LeakyReLU(),
            #nn.Conv2d(32, 64, 3), # 3x6
            #nn.BatchNorm2d(64),
            #nn.LeakyReLU(),
            #nn.Conv2d(64, 128, 3), # 3x6
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 8, 3), # 5x8 -> 4x7
            nn.MaxPool2d(2, 2),
            #nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 3), # 3x6
            nn.MaxPool2d(2, 2),
            #nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            #nn.Conv2d(16, 32, 3), # 3x6
            #nn.MaxPool2d(2, 2),
            #nn.BatchNorm2d(32),
            #nn.LeakyReLU(),
            #nn.Conv2d(32, 64, 3), # 3x6
            #nn.BatchNorm2d(64),
            #nn.LeakyReLU(),
            #nn.Conv2d(64, 128, 3), # 3x6
        )
        self.dconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=(7, 8)),  # 3x6 -> 5x8
            #nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=(5, 8)),  # 5x8 -> 9x16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=(5, 7)),  # 13x22
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=(4, 7)),  # 17x28
            #nn.LeakyReLU(),
            #nn.ConvTranspose2d(4, 1, kernel_size=(4, 7)),  # 21x34
            #nn.LeakyReLU(),
            #nn.ConvTranspose2d(1, 1, kernel_size=(5, 7)),  # 25x40
        )
        '''
        icra_map = Map(40, 25)
        grid = icra_map.getGrid()
        grid = 1 - grid
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid = torch.from_numpy(grid).to(device).double()
        self.device = device
        '''

    def forward(self, s):
        #print(s.shape)
        window_map = s[:,:1,:,:]
        enemy_map = s[:,1:,:,:]
        feature_map_1 = self.conv1(window_map)
        feature_map_2 = self.conv2(enemy_map)
        #feature_map = torch.cat([feature_map_1, feature_map_2], dim=1)
        feature_map = feature_map_1*1e-8 +  feature_map_2
        #batch, channel, w, h = feature_map.shape
        #feature_map = feature_map.reshape([batch, -1])
        #feature_map = self.fc(feature_map)
        #feature_map = feature_map.reshape([batch, channel, w, h])
        value_map = self.dconv(feature_map)
        #print(value_map.shape)
        #value_map = value_map + s[:,0:1,:,:]

        #batch_size = s.size(0)
        #feature = self.conv(m).squeeze(2).squeeze(1)
        #feature_map = self.fc(torch.cat([feature, s], dim=1))
        #feature_map = self.fc(s)
        #feature_map = feature_map.reshape(batch_size, 1, 5, 8)
        #value_map = self.dconv(feature_map)
        return value_map

if __name__ == "__main__":
    net = DQN()
    inputs = Variable(torch.rand([1,2,5,8]), requires_grad=True)
    #inputs2 = Variable(torch.rand([1,2,5,8]), requires_grad=True)
    out = net(inputs)
    #out = 2*inputs + 3*inputs2
    sss = out.sum()
    sss.backward()
    print(inputs.grad)
