from models import Classifier
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main(batch_size=64, epochs=10):
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
        ])),
        batch_size=batch_size, shuffle=True,
    )

    model = Classifier()
    summary(model, (1, 28, 28))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(loss.item())


if __name__ == '__main__':
    main()
