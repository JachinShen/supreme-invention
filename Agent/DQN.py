import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(10, 40),
            nn.LeakyReLU(),
        )
        #self.conv = nn.Sequential(
            #nn.Conv2d(1, 1, 3), # 5x8 -> 4x7
            #nn.LeakyReLU(),
            #nn.Conv2d(1, 1, 3), # 3x6
            #nn.LeakyReLU(),
            #nn.Conv2d(1, 1, 3), # 2x5
            #nn.LeakyReLU(),
            #nn.Conv2d(1, 1, 3), # 1x4
            #nn.LeakyReLU(),
        #)
        self.dconv = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(5, 9)),  # 5x8 -> 9x16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=(5, 7)),  # 13x22
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=(5, 7)),  # 17x28
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=(5, 7)),  # 21x34
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=(5, 7)),  # 25x40
        )

    def forward(self, s):
        batch_size = s.size(0)
        #feature = self.conv(m).squeeze(2).squeeze(1)
        #feature_map = self.fc(torch.cat([feature, s], dim=1))
        feature_map = self.fc(s)
        feature_map = feature_map.reshape(batch_size, 1, 5, 8)
        value_map = self.dconv(feature_map)
        return value_map
