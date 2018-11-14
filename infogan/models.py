import torch
import torch.nn as nn
import constants


class G(nn.Module):
    def __init__(self,):
        super(G, self).__init__()
        noise_dim = constants.NOISE_DIM
        categorical_dim = constants.CATEGORICAL_DIM

        self.fc_block = nn.Sequential(
            nn.Linear(noise_dim+categorical_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128*7*7),
            nn.BatchNorm1d(128*7*7),
            nn.ReLU(),
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, z, c):
        x = torch.cat((z, c), -1)
        x = self.fc_block(x)
        x = x.view(x.shape[0], 128, 7, 7)
        x = self.conv_blocks(x)

        return x


class FE(nn.Module):
    def __init__(self, ):
        super(FE, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc_blocks = nn.Sequential(
            nn.Linear(128*3*3, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_blocks(x)

        return x


class D(nn.Module):
    def __init__(self, ):
        super(D, self).__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc_block(x)

        return x


class Q(nn.Module):
    def __init__(self, ):
        super(Q, self).__init__()
        categorical_dim = constants.CATEGORICAL_DIM

        self.fc_block = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, categorical_dim)
        )
    
    def forward(self, x):
        x = self.fc_block(x)

        return x
