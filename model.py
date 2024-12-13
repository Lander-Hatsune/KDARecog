import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class Net(nn.Module):

    def __init__(self, confidence_thr=0.75):
        super(Net, self).__init__()
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(15 * 9, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 13)
        )
        self.softmax = nn.Softmax(dim=1)
        self.confidence_thr = confidence_thr


    def forward(self, x):
        x = self.flat(x)
        return self.fc(x)
    
    
    def pred(self, x):
        # Get logits from the forward pass
        logits = self.forward(x)
        probabilities = self.softmax(logits)
        
        # Calculate confidence (max probability for each sample in the batch)
        confidence, predictions = torch.max(probabilities, dim=1)
        
        # Apply confidence threshold logic
        predictions[confidence < self.confidence_thr] = 10  # "evaluation failure" token
        
        return predictions, confidence
