from models import G, FE, D, Q
from utils import weights_init_normal, to_categorical
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from torchsummary import summary
import constants


def main():
    noise_dim = constants.NOISE_DIM
    categorical_dim = constants.CATEGORICAL_DIM
    batch_size = constants.BATCH_SIZE
    epochs = constants.EPOCHS
    q_regularize = constants.Q_REGULARIZE

    if torch.cuda.is_available():
        device = 'cuda'
        print('GPU MODE')
    else:
        device = 'cpu'
        print('CPU MODE')
    device = torch.device(device)

    g = G().to(device)
    fe = FE().to(device)
    d = D().to(device)
    q = Q().to(device)

    adversarial_loss = torch.nn.BCELoss()
    categorical_loss = torch.nn.CrossEntropyLoss()

    g.apply(weights_init_normal)
    fe.apply(weights_init_normal)
    d.apply(weights_init_normal)
    q.apply(weights_init_normal)

    os.makedirs('./data/mnist', exist_ok=True)
    dataloader = DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])),
        batch_size=batch_size, shuffle=True
    )

    optim_GQ = torch.optim.Adam(list(g.parameters()) + list(q.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(list(fe.parameters()) + list(d.parameters()), lr=0.0002, betas=(0.5, 0.999))

    loss_Ds = []
    loss_Gs = []
    loss_Qs = []

    fixed_noise = torch.randn((batch_size, noise_dim), dtype=torch.float32, device=device)
    fixed_label = np.array([[i] * 10 for i in range(10)]).reshape(-1,)
    fixed_categorical = to_categorical(fixed_label).to(device, dtype=torch.float32)

    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device, dtype=torch.float32)

            real = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
            fake = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)

            # train D, real
            optim_D.zero_grad()

            fe_real = fe(imgs)
            a_real = d(fe_real)
            loss_real = adversarial_loss(a_real, real)

            # train D, fake
            noise = torch.randn((batch_size, noise_dim), dtype=torch.float32, device=device)
            label = np.random.randint(0, categorical_dim, batch_size)
            categorical = to_categorical(label).to(device, dtype=torch.float32)

            gen_imgs = g(noise, categorical)
            fe_fake = fe(gen_imgs.detach())
            a_fake = d(fe_fake)
            loss_fake = adversarial_loss(a_fake, fake)

            # update D
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optim_D.step()

            # train G and Q
            optim_GQ.zero_grad()

            noise = torch.randn((batch_size, noise_dim), dtype=torch.float32, device=device)
            label = np.random.randint(0, categorical_dim, batch_size)
            categorical = to_categorical(label).to(device, dtype=torch.float32)

            gen_imgs = g(noise, categorical)
            fe_trick = fe(gen_imgs)
            a_trick = d(fe_trick)
            q_trick = q(fe_trick)

            loss_G = adversarial_loss(a_trick, real)

            label = torch.from_numpy(label).to(device, dtype=torch.long)
            loss_Q = categorical_loss(q_trick, label)

            # update G and Q
            loss_GQ = loss_G + loss_Q * q_regularize
            loss_GQ.backward()
            optim_GQ.step()

            loss_Ds.append(loss_D.item())
            loss_Gs.append(loss_G.item())
            loss_Qs.append(loss_Q.item())

            if i % 100 == 0:
                print('Epoch/Iter:{0:4}/{1:4}, loss_D: {2:.3f}, loss_G: {3:.3f}, loss_Q: {4:.3f}'.format(
                    epoch, i, loss_Ds[-1], loss_Gs[-1], loss_Qs[-1],
                ))
        
        torch.save(g.state_dict(), './log/g_epoch{:04}.pth'.format(epoch))
        torch.save(fe.state_dict(), './log/fe_epoch{:04}.pth'.format(epoch))
        torch.save(d.state_dict(), './log/d_epoch{:04}.pth'.format(epoch))
        torch.save(q.state_dict(), './log/q_epoch{:04}.pth'.format(epoch))

        fixed_gen_imgs = g(fixed_noise, fixed_categorical)
        torchvision.utils.save_image(fixed_gen_imgs.detach(), './log/image_epoch{:04}.png'.format(epoch), normalize=True, nrow=10)
    
    plt.plot(loss_Ds, label='D')
    plt.plot(loss_Gs, label='G')
    plt.plot(loss_Qs, label='Q')
    plt.legend()
    plt.savefig('./model/sagan/plot_loss_through_iterations.png')


if __name__ == '__main__':
    os.makedirs('./log', exist_ok=True)
    main()



