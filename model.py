import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class Net(nn.Module):

    def __init__(self, confidence_thr=0.7):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (15x9 -> 15x9)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                # (15x9 -> 7x4)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (7x4 -> 7x4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                 # (7x4 -> 3x2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 13)
        )
        self.softmax = nn.Softmax(dim=1)
        self.confidence_thr = confidence_thr


    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

    
    def pred(self, x):
        # Get logits from the forward pass
        logits = self.forward(x)
        probabilities = self.softmax(logits)
        
        # Calculate confidence (max probability for each sample in the batch)
        confidence, predictions = torch.max(probabilities, dim=1)
        
        # Apply confidence threshold logic
        predictions[confidence < self.confidence_thr] = 10  # "evaluation failure" token
        
        return predictions, confidence
