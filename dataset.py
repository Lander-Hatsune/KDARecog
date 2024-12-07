import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class KDADataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with the dataset in /data/{label}/{image}.png format.
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Walk through the directory and collect image paths and labels
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_file)
                    self.data.append((img_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

