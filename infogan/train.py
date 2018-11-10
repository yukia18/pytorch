from models import G, FrontEnd, D, Q
from utils import weights_init_normal, to_categorical
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import itertools
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from torchsummary import summary


NOISE_DIM = 50
LABEL_DIM = 10
EPOCHS = 2


def main():
    input_dim = NOISE_DIM + LABEL_DIM

    g = G(input_dim=input_dim)
    fe = FrontEnd()
    d = D()
    q = Q()

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
        batch_size=100, shuffle=True
    )

    opt_G = torch.optim.Adam([{'params': g.parameters()}, {'params': q.parameters()}], lr=0.0002, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam([{'params': fe.parameters()}, {'params': d.parameters()}], lr=0.0002, betas=(0.5, 0.999))

    loss_Ds = []
    loss_Gs = []
    loss_Qs = []

    for epoch in range(EPOCHS):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]

            real_imgs = Variable(imgs.type(torch.FloatTensor))

            real = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # train D, real
            opt_D.zero_grad()

            fe_real = fe(real_imgs)
            a_real = d(fe_real)
            loss_real = adversarial_loss(a_real, real)
            loss_real.backward()

            # train D, fake
            noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, NOISE_DIM))))
            label = np.random.randint(0, LABEL_DIM, batch_size)
            label_ = to_categorical(label, num_columns=LABEL_DIM)

            gen_imgs = g(noise, label_)
            fe_fake = fe(gen_imgs)
            a_fake = d(fe_fake)
            loss_fake = adversarial_loss(a_fake, fake)
            loss_fake.backward()

            loss_D = loss_real + loss_fake / 2
            opt_D.step()

            # train G and Q
            opt_G.zero_grad()

            fe_out = fe(gen_imgs.detach())
            a_trick = d(fe_out)
            loss_trick = adversarial_loss(a_trick, real)

            q_logit = q(fe_out)
            label_ = Variable(torch.LongTensor(label))
            loss_info = categorical_loss(q_logit, label_)

            loss_G = loss_trick + loss_info
            loss_G.backward()
            opt_G.step()

            if i % 100 == 0:
                print('Epoch/Iter:{0:4}/{1:4}, loss_D: {2:.3f}, loss_G: {3:.3f}'.format(
                    epoch, i, loss_D.data.numpy(), loss_G.data.numpy())
                )


if __name__ == '__main__':
    main()



