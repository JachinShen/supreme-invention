import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple

import sys
sys.path.append(".")

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        h_dim = 256*1*1
        z_dim = 64
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(4,4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(4,4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(4,4), stride=2, padding=1),
            nn.ReLU(),
            #nn.Conv2d(128, 256, kernel_size=(4,4), stride=2, padding=1),
            #nn.ReLU(),
            #nn.Conv2d(256, 512, kernel_size=(4,4), stride=2, padding=1),
            #nn.ReLU(),
            nn.Conv2d(128, 4000, 1)
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4000, 256, kernel_size=(4,4), stride=2, padding=1),
            nn.ReLU(),
            #nn.ConvTranspose2d(512, 256, kernel_size=(4,4), stride=2, padding=1),
            #nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4,3), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(6,4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1, 1, 1),
            nn.Tanh(),
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device).double()
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        print(x.shape)
        h = self.encoder(x)
        print(h.shape)
        #batch, channel, width, height = h.shape
        #h = h.reshape([batch, -1])
        #z, mu, logvar = self.bottleneck(h)
        #z = self.fc1(h)
        #z = self.fc3(z)
        
        #z = z.reshape([batch, channel, width, height])
        restore = self.decoder(h)
        print(restore.shape)
        return restore

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

if __name__ == "__main__":
    BATCH_SIZE = 512
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device).double()
    optimizer = torch.optim.Adam(model.parameters())
    memory = torch.load("replay.memory", map_location=device)
    for epoch in range(2000):
        print("Epoch: [{}/{}]".format(epoch, 2000))
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(device).double()[:,:2,:,:]
        y = model(state_batch)
        loss = nn.MSELoss()(state_batch, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print("Loss: {}".format(loss.item()))
            torch.save(model.state_dict(), "VAE.model")
