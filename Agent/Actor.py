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
        self.fc = nn.Sequential(
            nn.Linear(36*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )
        self.head_a = nn.Sequential(
            nn.Linear(256, 6), # 8x4
            nn.Softmax(dim=1),
        )
        self.head_v = nn.Sequential(
            nn.Linear(256, 1),
        )

    def forward(self, s):
        batch, channel, seq = s.shape
        s = s.reshape([batch, -1])
        head = self.fc(s)
        a, v = self.head_a(head), self.head_v(head)
        return a, v

if __name__=="__main__":
    x = torch.rand([1,2,36])
    model = ActorCritic()
    y = model(x)
    print(y)