import torch
import torch.nn as nn


class G(nn.Module):
    def __init__(self, input_dim, ):
        super(G, self).__init__()

        self.l1 = nn.Linear(input_dim, 64*7*7)
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 5, stride=2, padding=2, output_padding=1),
            nn.Tanh(),
        )
    
    def forward(self, noise, labels):
        gen_input = torch.cat((noise, labels), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 64, 7, 7)
        img = self.conv_blocks(out)

        return img


class FrontEnd(nn.Module):
    def __init__(self, ):
        super(FrontEnd, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5),
            nn.AvgPool2d(7),
        )
    
    def forward(self, img):
        x = self.conv_blocks(img)
        fuse = x.view(x.shape[0], -1)

        return fuse


class D(nn.Module):
    def __init__(self, ):
        super(D, self).__init__()

        self.la = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, fuse):
        a = self.la(fuse)

        return a


class Q(nn.Module):
    def __init__(self, ):
        super(Q, self).__init__()

        self.lq = nn.Sequential(
            nn.Linear(64, 10),
        )
    
    def forward(self, fuse):
        q = self.lq(fuse)

        return q
