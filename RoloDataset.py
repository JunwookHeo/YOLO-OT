import glob
import os
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader


class RoloDataset(Dataset):
    """ Loading frames in a video file """
    def __init__(self, path, label, seq_num, img_size):
        self.path = path
        self.label = label
        self.seq_num = seq_num
        self.img_size = img_size
        
        self.frames = sorted(glob.glob("%s/*.*" % os.path.join(path, 'images'))) 
        self.images= sorted(glob.glob("%s/*.*" % os.path.join(path, 'yot_out')))
        
        with open(label, "r") as file:
            self.labels = file.readlines()

        self.num_frames = len(self.images)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        frame = np.array(Image.open(self.frames[idx]))
        image = np.load(self.images[idx])
        image = torch.from_numpy(image)
        fi = image[0:128*52*52].reshape(128, 52, 52)
        loc = image[128*52*52:]
        #label = self.labels[idx]
        label = self.labels[idx].split('\t')   # for gt type 2
        if len(label) < 4:
            label = self.labels[idx].split(',') # for gt type 1
        
        label = torch.as_tensor(np.array(label, dtype=int), dtype=torch.float32)

        return frame, fi, loc, label
