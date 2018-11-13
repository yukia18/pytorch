import torch
import torch.nn as nn
from torchvision.models import vgg16


class LossNet(nn.Module):
    def __init__(self):
        super(LossNet, self).__init__()
        self.features = vgg16(pretrained=True).features
        self.fetch_layers = {'3', '8', '15', '22'}

    def forward(self, x):
        output = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.fetch_layers:
                output.append(x)
        
        return output


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.features = vgg16(pretrained=True).features
        for params in self.features.parameters():
            params.requires_grad = False
        self.gap = nn.AvgPool2d(3)
        self.fc = nn.Linear(512, 10)
    
    def forward(self, img):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


class Attacker(nn.Module):
    def __init__(self):
        super(Attacker, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 96, 5, stride=1, padding=2),
            nn.Conv2d(96, 96, 5, stride=1, padding=2),
            nn.Conv2d(96, 3, 5, stride=1, padding=2),
        )
    
    def forward(self, img):
        x = self.conv_block(img)
        x += img
        x = nn.Sigmoid(x)

        return x