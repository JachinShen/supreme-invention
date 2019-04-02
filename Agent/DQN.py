import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
#import sys
#sys.path.append(".")
#from util.Grid import Map

#from Referee.ICRAMap import ICRAMap

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        '''
        self.fc = nn.Sequential(
            nn.Linear(16*3*6, 64),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 16*3*6),
        )
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 2), # 5x8 -> 4x7
            nn.LeakyReLU(),
            nn.Conv2d(4, 16, 2), # 3x6
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 2), # 3x6
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 2), # 3x6
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 4, 2), # 5x8 -> 4x7
            nn.LeakyReLU(),
            nn.Conv2d(4, 16, 2), # 3x6
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 2), # 3x6
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 2), # 3x6
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 4, 2), # 5x8 -> 4x7
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

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.map = ICRAMap().image.reshape(1, 1, 50, 80)
        self.map = torch.from_numpy(self.map).double().to(device)
        '''

    def encode(self, s):
        window_map = s[:,0:1,:,:]
        enemy_map = s[:,1:2,:,:]
        last_value_map = s[:,2:3,:,:]
        feature_map_1 = self.conv1(window_map)
        feature_map_2 = self.conv2(enemy_map)
        feature_map = feature_map_1 + feature_map_2
        return feature_map


    def forward(self, s):
        window_map = s[:,0:1,:,:]
        enemy_map = s[:,1:2,:,:]
        last_value_map = s[:,2:3,:,:]
        feature_map_1 = self.conv1(window_map)
        feature_map_2 = self.conv2(enemy_map)
        #feature_map_3 = self.conv3(last_value_map)
        #feature_map = torch.cat([feature_map_1, feature_map_2, ], dim=1)
        #feature_map = feature_map_1 + feature_map_2 + feature_map_3
        feature_map = feature_map_1 + feature_map_2

        #batch, channel, w, h = feature_map.shape
        #feature_map = feature_map.reshape([batch, -1])
        #feature_map = self.fc(feature_map)
        #feature_map = feature_map.reshape([batch, channel, w, h])

        value_map = self.dconv(feature_map)
        batch, channel, w, h = value_map.shape
        value_map = value_map.reshape([batch, -1])
        value_map = F.softmax(value_map, dim=1)
        value_map = value_map.reshape([batch, channel, w, h])

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
