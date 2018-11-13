import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),
            nn.AvgPool2d(7),
        )
        self.fc_block = nn.Sequential(
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_block(x)

        return x