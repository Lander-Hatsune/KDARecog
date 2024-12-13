import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
from dataset import KDADataset
from model import Net

torch.manual_seed(814)

EPOCH = 300
NUM_CLASSES = 13  # Number of classes in your dataset

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="./logs")

train_transform = transforms.Compose([
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
train_dataset = KDADataset(root_dir="data13-ck-v6/train", transform=train_transform)
validate_dataset = KDADataset(root_dir="data13-ck-v6/validate", transform=val_transform)

train_dataloader = DataLoader(train_dataset,
                              batch_size=64, shuffle=True)

validate_dataloader = DataLoader(validate_dataset,
                                 batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    # Log class-specific accuracies to TensorBoard
    class_accuracy = class_correct / class_total
    # Log class-specific accuracies to a single graph in TensorBoard
    for i in range(NUM_CLASSES):
        accuracy = class_accuracy[i].item() * 100 if class_total[i] > 0 else 0.0
        writer.add_scalars('Validation/Class_Accuracies', {f'Class_{i}': accuracy}, epoch)

    # Calculate and log overall validation accuracy
    overall_validate_acc = class_correct.sum() / class_total.sum()
    writer.add_scalar('Validation/Overall_Accuracy', overall_validate_acc, epoch)
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
    
    # Log training loss to TensorBoard
    writer.add_scalar('Training/Loss', train_loss, epoch)
    print(f'Training loss for Epoch {epoch + 1}: {train_loss:.7f}\n')

# Save final model
torch.save(net.state_dict(), 'Net.pt')
writer.close()  # Close TensorBoard writer
