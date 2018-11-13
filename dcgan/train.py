from models import Generator, Discriminator
from utils import weights_init_normal
import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt


def main(batch_size=100, noise_dim=100, epochs=100):
    if torch.cuda.is_available():
        device = 'cuda'
        print('GPU MODE')
    else:
        device = 'cpu'
        print('CPU MODE')
    device = torch.device(device)

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

    optimizerG = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    loss_Ds = []
    loss_Gs = []

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device, dtype=torch.float32)

            real = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
            fake = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)

            # train D, real
            optimizerD.zero_grad()

            a_real = discriminator(imgs)
            loss_real = criterion(a_real, real)

            # train D, fake
            noise = torch.randn((batch_size, noise_dim), dtype=torch.float32, device=device)

            gen_imgs = generator(noise)
            a_fake = discriminator(gen_imgs.detach())
            loss_fake = criterion(a_fake, fake)

            # update D
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizerD.step()

            # train G
            optimizerG.zero_grad()

            noise = torch.randn((batch_size, noise_dim), dtype=torch.float32, device=device)

            trick_imgs = generator(noise)
            a_trick = discriminator(trick_imgs)
            loss_G = criterion(a_trick, real)

            # update G
            loss_G.backward()
            optimizerG.step()

            loss_Ds.append(loss_D.item())
            loss_Gs.append(loss_G.item())

            if i % 100 == 0:
                print('Epoch/Iter:{0:4}/{1:4}, loss_D: {2:.3f}, loss_G: {3:.3f}'.format(
                    epoch, i, loss_Ds[-1], loss_Gs[-1],
                ))
        
        torch.save(generator.state_dict(), './log/generator_epoch{:04}.pth'.format(epoch))
        torch.save(discriminator.state_dict(), './log/discriminator_epoch{:04}.pth'.format(epoch))

        gen_imgs = generator(fixed_noise)
        torchvision.utils.save_image(gen_imgs.detach(), './log/image_epoch{:04}.png'.format(epoch), normalize=True)
    
    plt.plot(loss_Ds, label='Discriminator')
    plt.plot(loss_Gs, label='Generator')
    plt.legend()
    plt.savefig('./log/plot_loss_through_iterations.png')


if __name__ == '__main__':
    os.makedirs('./log', exist_ok=True)
    main()
