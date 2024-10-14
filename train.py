import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import KDADataset
from model import Net

torch.manual_seed(814)

EPOCH = 100

train_dataset = KDADataset('data13-ck-v4/train')
train_dataloader = DataLoader(train_dataset,
                              batch_size=64, shuffle=True)

validate_dataset = KDADataset('data13-ck-v4/validate')
validate_dataloader = DataLoader(validate_dataset,
                                 batch_size=64, shuffle=False)

device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')

net = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

best_validate_acc = 0

for epoch in range(EPOCH):
    print(f'epoch: {epoch}')
    validate_cnt = 0
    validate_correct_cnt = 0
    for batchcnt, (imgs, labels) in enumerate(validate_dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        validate_cnt += len(labels)
        with torch.no_grad():
            preds = net(imgs)
        validate_correct_cnt += torch.sum(preds.max(1)[1] == labels)
    validate_acc = validate_correct_cnt / validate_cnt
    print(f'validate acc: {validate_acc:.4f}')

    if validate_acc > best_validate_acc:
        best_validate_acc = validate_acc
        torch.save(net.state_dict(), 'Net.pt')

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

