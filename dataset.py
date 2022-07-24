import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class KDADataset(Dataset):

    def __init__(self):

        self.pathlist = []
        self.labels = []
        for i in range(11):
            for filename in os.listdir(f'data/{i}/'):
                self.pathlist.append(f'data/{i}/{filename}')
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)
                
    def __getitem__(self, i):
        image = torch.tensor(np.array(Image.open(self.pathlist[i])) / 255,
                             dtype=torch.float)
        return image, self.labels[i]
        
