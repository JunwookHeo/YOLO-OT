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
        return self.num_frames - (self.seq_num - 1) 

    def __getitem__(self, idx):
        frames = []
        fis = []
        locs = []
        labels = []
        
        for i in range(self.seq_num):
            pos = idx + i
            frame = np.array(Image.open(self.frames[pos]))
            frame = torch.from_numpy(frame)

            image = np.load(self.images[pos])
            image = torch.from_numpy(image)
            fi = image[0:128*52*52].reshape(128, 52, 52)
            loc = image[128*52*52:]
            #label = self.labels[pos]
            label = self.labels[pos].split('\t')   # for gt type 2
            if len(label) < 4:
                label = self.labels[pos].split(',') # for gt type 1
            
            label = torch.as_tensor(np.array(label, dtype=int), dtype=torch.float32)
            
            frames.append(frame)
            fis.append(fi)
            locs.append(loc)
            labels.append(label)
        
        return torch.stack(frames, dim=0), torch.stack(fis, dim=0), torch.stack(locs, dim=0), torch.stack(labels, dim=0)
