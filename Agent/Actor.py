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
        self.extractor = nn.Sequential(
            nn.Conv1d(2, 16, 6), # 135 -> 130
            nn.MaxPool1d(2, 2), # 65
            nn.ReLU(),
            nn.Conv1d(16, 64, 6), # 60
            nn.MaxPool1d(2, 2), # 30
            nn.ReLU(),
            nn.Conv1d(64, 128, 3), # 28
            nn.MaxPool1d(2, 2), # 14
            nn.ReLU(),
            nn.Conv1d(128, 256, 3), # 12
            nn.MaxPool1d(2, 2), # 6
            nn.ReLU(),
            nn.Conv1d(256, 512, 3), # 4
            nn.MaxPool1d(2, 2), # 2
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(512*2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.head_a = nn.Sequential(
            nn.Linear(128, 6), # 8x4
            nn.Softmax(dim=1),
        )
        self.head_v = nn.Sequential(
            nn.Linear(128, 1),
        )

    def forward(self, s):
        h = self.extractor(s)
        #print(h.shape)
        batch, channel, seq = h.shape
        h = h.reshape([batch, -1])
        head = self.fc(h)
        a, v = self.head_a(head), self.head_v(head)
        return a, v

if __name__=="__main__":
    x = torch.rand([1,2,135])
    model = ActorCritic()
    y = model(x)
    print(y)