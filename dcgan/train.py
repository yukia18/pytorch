from models import Generator, Discriminator
from utils import weights_init_normal
import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import os


def main(batch_size=100, noise_dim=100, epochs=300, device='cpu'):
    dataloader = DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])),
        batch_size=batch_size, shuffle=True,
    )

    generator = Generator(noise_dim).to(device)
    discriminator = Discriminator().to(device)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    summary(generator, (noise_dim, ))
    summary(discriminator, (1, 28, 28))

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(batch_size, noise_dim, device=device)

    optimizerG = torch.optim.Adam([{'params': generator.parameters()},], lr=0.0002, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam([{'params': discriminator.parameters()},], lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)

            discriminator.zero_grad()

            real = torch.Tensor(batch_size, 1).fill_(1.0).to(device)
            fake = torch.Tensor(batch_size, 1).fill_(0.0).to(device)

            a_real = discriminator(imgs)
            loss_real = criterion(a_real, real)
            loss_real.backward()

            noise = torch.randn(batch_size, noise_dim).to(device)
            gen_imgs = generator(noise)
            a_fake = discriminator(gen_imgs.detach())
            loss_fake = criterion(a_fake, fake)
            loss_fake.backward()

            loss_D = loss_real + loss_fake
            optimizerD.step()

            generator.zero_grad()
            a_trick = discriminator(gen_imgs)
            loss_G = criterion(a_trick, real)
            loss_G.backward()
            optimizerG.step()

            if i % 100 == 0:
                print('Epoch/Iter:{0:4}/{1:4}, loss_D: {2:.3f}, loss_G: {3:.3f}'.format(
                    epoch, i, loss_D.data.numpy(), loss_G.data.numpy())
                )

        gen_imgs = generator(fixed_noise)
        torchvision.utils.save_image(gen_imgs.detach(), './log/generated_image_epoch{:4}.png'.format(epoch), normalize=True)
        torch.save(generator.state_dict(), './log/generator_epoch{:4}.pth'.format(epoch))
        torch.save(discriminator.state_dict(), './log/discriminator_epoch{:4}.pth'.format(epoch))


if __name__ == '__main__':
    os.makedirs('./log', exist_ok=True)
    main(device='cuda:0')