import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Referee.ICRAMap import ICRAMap

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(9, 25),
        #     nn.LeakyReLU(),
        # )
        #
        # self.dconv = nn.Sequential(
        #     nn.ConvTranspose2d(1, 1, kernel_size=(3,3)), # 5x5 -> 7x7
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose2d(1, 1, kernel_size=(3,3)), # 7x7 -> 9x9
        # )

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 3), stride=2, padding=1), #50*80 -> 25*40*4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), padding=1), #25*40*4 -> 9*14*4
            nn.Conv2d(4, 10, kernel_size=(2, 2), stride=2, padding=0), #9*14*4 -> 4*7*10
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(280, 25),
            nn.LeakyReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(35, 16),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.sigmoid_sj = nn.Sequential(
            nn.Linear(24, 3),
            nn.Sigmoid()
        )

        self.fc4_tu = nn.Sequential(
            nn.Linear(24, 25),
            nn.ReLU()
        )

        self.dconv_tu = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(3, 3)),  # 5x5 -> 7x7
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1, 3, kernel_size=(3, 3)),  # 7x7 -> 9x9
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.map = ICRAMap().image.reshape(1, 1, 50, 80)
        self.map = torch.from_numpy(self.map).double().to(device)

    def forward(self, s):
        batch_size = s.size(0)
        state_num = s.size(1)
        # print(batch_size)

        state = torch.squeeze(s).view(batch_size, state_num) #b * 10

        #SJ
        x_sj = self.cnn(self.map).view(-1) #280
        map_feature_sj = self.fc1(x_sj) #25
        map_feature_sj = map_feature_sj.repeat(batch_size).reshape(batch_size, 25) #b * 25
        #map_feature_sj = self.fc1(self.cnn(self.map).view(batch_size, -1)) #b * 25
        x_sj = torch.cat((map_feature_sj, state), 1) # b * 35
        x_sj = self.fc2(x_sj) # b * 16
        y_sj = self.fc3(x_sj) # b * 8
        x_sj = self.sigmoid_sj(torch.cat((x_sj, y_sj), 1)) # b* 3

        #TU
        x_tu = self.cnn(self.map).view(-1)  # 280
        map_feature_tu = self.fc1(x_tu)  # 25
        map_feature_tu = map_feature_tu.repeat(batch_size).reshape(batch_size, 25)  # b * 25
        # map_feature_tu = self.fc1(self.cnn(self.map).view(batch_size, -1)) #b * 25
        x_tu = self.fc2(torch.cat((map_feature_tu, state), 1)) # b * 16
        y_tu = self.fc3(x_tu) #b * 8
        x_tu = self.fc4_tu(torch.cat((x_tu, y_tu), 1)) # b * 25
        x_tu = x_tu.reshape(batch_size, 1, 5, 5) # b * 1 * 5 * 5
        x_tu = self.dconv_tu(x_tu) # b * 3 * 9 * 9

        #merge
        value_map = x_tu[:,torch.max(x_sj, 1)[1],:,:]

        return value_map
















        #
        # feature_map = self.fc(s)
        # feature_map = feature_map.reshape(batch_size, 1, 5, 5)
        # value_map = self.dconv(feature_map)
        # return value_map
