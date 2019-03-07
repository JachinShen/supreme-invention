import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(9, 25),
            nn.LeakyReLU(),
        )
        self.dconv = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(3,3)), # 5x5 -> 7x7
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=(3,3)), # 7x7 -> 9x9
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=(3,3)), # 11x11
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=(3,3)), # 13x13
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=(3,3)), # 15x15
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=(3,3)), # 17x17
        )

    def forward(self, s):
        batch_size = s.size(0)
        feature_map = self.fc(s)
        feature_map = feature_map.reshape(batch_size, 1, 5, 5)
        value_map = self.dconv(feature_map)
        return value_map
