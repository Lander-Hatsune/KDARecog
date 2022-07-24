import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(14 * 9, 128),
            nn.ReLU(),
            nn.Linear(128, 12))

    def forward(self, x):

        x = self.flat(x)
        return self.fc(x)
