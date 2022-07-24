import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import KDADataset
from model import Net

torch.manual_seed(814)

EPOCH = 100

train_dataloader = DataLoader(KDADataset('data/data_train'),
                              batch_size=64, shuffle=True)
validate_dataloader = DataLoader(KDADataset('data/data_validate'),
                                 batch_size=240, shuffle=False)

device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')

net = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

for epoch in range(EPOCH):
    print(f'epoch: {epoch}')
    for batchcnt, (imgs, labels) in enumerate(validate_dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            preds = net(imgs)
    print(f'validate acc: {torch.sum(preds.max(1)[1] == labels) / 240:.4f}')
    train_loss = 0
    for batchcnt, (imgs, labels) in enumerate(train_dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = net(imgs)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f'train loss: {train_loss:.7f}')

torch.save(net.state_dict(), 'Net.pt')
