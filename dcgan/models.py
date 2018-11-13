import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_dim, 128*7*7)
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise):
        x = self.fc(noise)
        x = x.view(x.shape[0], 128, 7, 7)
        x = self.conv_block(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),
        )
        self.fc_block = nn.Sequential(
            nn.Linear(6272, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        x = self.conv_block(img)
        x = x.view(x.shape[0], -1)
        x = self.fc_block(x)

        return x