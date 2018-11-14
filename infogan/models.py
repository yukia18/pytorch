import torch
import torch.nn as nn
import constants


class G(nn.Module):
    def __init__(self,):
        super(G, self).__init__()
        noise_dim = constants.NOISE_DIM
        categorical_dim = constants.CATEGORICAL_DIM

        self.fc_block = nn.Sequential(
            nn.Linear(noise_dim+categorical_dim, 64*7*7),
            nn.ReLU(),
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 5, stride=2, padding=2, output_padding=1),
            nn.Tanh(),
        )
    
    def forward(self, z, c):
        x = torch.cat((z, c), -1)
        x = self.fc_block(x)
        x = x.view(x.shape[0], 64, 7, 7)
        x = self.conv_blocks(x)

        return x


class FE(nn.Module):
    def __init__(self, ):
        super(FrontEnd, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),
        )
    
    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.shape[0], -1)

        return x


class D(nn.Module):
    def __init__(self, ):
        super(D, self).__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(64*7*7, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc_block(x)

        return x


class Q(nn.Module):
    def __init__(self, ):
        super(Q, self).__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(64*7*7, 10),
        )
    
    def forward(self, x):
        x = self.fc_block(x)

        return x
