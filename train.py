import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import KDADataset
from model import Net

torch.manual_seed(814)

EPOCH = 100
NUM_CLASSES = 13  # Number of classes in your dataset

train_transform = transforms.Compose([
    # transforms.RandomRotation(degrees=15),  # Rotate by ±15 degrees
    # transforms.RandomHorizontalFlip(),      # 50% chance to flip
    transforms.RandomResizedCrop(size=(15, 9), scale=(0.8, 1.0)),  # Random crop and resize
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness and contrast
    transforms.ToTensor(),                  # Convert to Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# Validation Transform (No Augmentation)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Initialize datasets
train_dataset = KDADataset(root_dir="data12-ck-v1/train", transform=train_transform)
validate_dataset = KDADataset(root_dir="data12-ck-v1/validate", transform=val_transform)

train_dataloader = DataLoader(train_dataset,
                              batch_size=64, shuffle=True)

validate_dataloader = DataLoader(validate_dataset,
                                 batch_size=64, shuffle=False)

device = torch.device('cpu')  # 'cuda' if torch.cuda.is_available() else 'cpu'

net = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

best_validate_acc = 0

for epoch in range(EPOCH):
    print(f'======== EPOCH: {epoch + 1}/{EPOCH} ========')
    
    # Initialize counters for per-class accuracy tracking
    class_correct = torch.zeros(NUM_CLASSES).to(device)  # Correct predictions per class
    class_total = torch.zeros(NUM_CLASSES).to(device)    # Total samples per class
    
    net.eval()  # Set model to evaluation mode
    for batchcnt, (imgs, labels) in enumerate(validate_dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        
        with torch.no_grad():
            preds = net(imgs)
        predicted_labels = preds.max(1)[1]
        
        # Update class-specific counters
        for i in range(NUM_CLASSES):
            class_correct[i] += torch.sum((predicted_labels == i) & (labels == i))
            class_total[i] += torch.sum(labels == i)
    
    # Display class-specific accuracies
    print(f'Validation Results for Epoch {epoch + 1}')
    print(f"{'Class':^10} | {'Correct':^10} | {'Total':^10} | {'Accuracy (%)':^15}")
    print('-' * 50)
    
    class_accuracy = class_correct / class_total
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:  # Avoid division by zero
            accuracy = class_accuracy[i].item() * 100
            print(f'{i:^10} | {int(class_correct[i].item()):^10} | {int(class_total[i].item()):^10} | {accuracy:^15.2f}')
        else:
            print(f'{i:^10} | {"-":^10} | {"-":^10} | {"N/A":^15}')
    
    # Calculate and display overall validation accuracy
    overall_validate_acc = class_correct.sum() / class_total.sum()
    print(f'\nOverall validation accuracy: {overall_validate_acc:.4f}\n')

    # Save model if this is the best validation accuracy so far
    if overall_validate_acc > best_validate_acc:
        best_validate_acc = overall_validate_acc
        torch.save(net.state_dict(), 'Net.val.pt')
        print("Best model saved.\n")

    # Training phase
    net.train()  # Set model back to training mode
    train_loss = 0
    for batchcnt, (imgs, labels) in enumerate(train_dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = net(imgs)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f'Training loss for Epoch {epoch + 1}: {train_loss:.7f}\n')

torch.save(net.state_dict(), 'Net.pt')


