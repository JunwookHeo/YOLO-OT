import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class FramesDataset(Dataset):
    def __init__(self, path, label, seq_num, img_size):
        self.path = path
        self.label = label
        self.seq_num = seq_num
        self.img_size = img_size
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass
    def resize(self, frame, img_size):
        pass

