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
            nn.Linear(135*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )
        self.head_a_m = nn.Sequential(
            nn.Linear(256, 3), # 8x4
            nn.Softmax(dim=1),
        )
        self.head_a_t = nn.Sequential(
            nn.Linear(256, 3), # 8x4
            nn.Softmax(dim=1),
        )
        self.head_v = nn.Sequential(
            nn.Linear(256, 1),
        )

    def forward(self, s):
        #h = self.extractor(s)
        h = s
        shortcut = torch.stack([
            torch.mean(s[:,0,135//2+45//2:], dim=1), 
            torch.mean(s[:,0,135//2-45//2:135//2+45//2], dim=1), 
            torch.mean(s[:,0,:135//2-45//2], dim=1)], dim=1) # batch, 3
        shortcut = F.softmax(shortcut, dim=1)
        #print(h.shape)
        batch, channel, seq = h.shape
        h = h.reshape([batch, -1])
        head = self.fc(h)
        a_m, a_t, v = self.head_a_m(head), self.head_a_t(head), self.head_v(head)
        a_m = F.softmax(a_m + shortcut, dim=1)
        return a_m, a_t, v

if __name__=="__main__":
    x = torch.rand([1,2,135])
    model = ActorCritic()
    y = model(x)
    print(y)