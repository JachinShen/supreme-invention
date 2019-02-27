import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 160),
        )

    def forward(self, s):
        value_map = self.fc(s)
        value_map = value_map.reshape([-1, 16, 10])
        return value_map
