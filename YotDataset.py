import glob
import os
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from coord_utils import *

class YotDataset(Dataset):
    """ Loading frames in a video file """
    def __init__(self, path, label, seq_num, img_size, mode):
        self.path = path
        self.label = label
        self.seq_num = seq_num
        self.img_size = img_size
        self.mode = mode
                
        self.frames = sorted(glob.glob("%s/*.*" % os.path.join(path, 'images'))) 
        self.images= sorted(glob.glob("%s/*.*" % os.path.join(path, 'yot_out')))
        
        with open(label, "r") as file:
            self.labels = file.readlines()

        self.num_frames = len(self.images)
        self.start_pos = 0

        if self.mode is 'train':    
            self.num_frames = int(self.num_frames*0.6)
        elif self.mode is 'validate':
            self.start_pos = int(self.num_frames*0.6)
            end_pos = int(self.num_frames*0.8)
            self.num_frames = end_pos - self.start_pos        
        elif self.mode is 'test':
            self.start_pos = int(self.num_frames*0.8)
            self.num_frames = self.num_frames - self.start_pos
        
    def __len__(self):
        return self.num_frames - (self.seq_num - 1) 

    def __getitem__(self, idx):
        frames = []
        fis = []
        locs = []
        labels = []
        
        for i in range(self.seq_num):
            pos = idx + i + self.start_pos
            frame = np.array(Image.open(self.frames[pos]))
            frame = torch.from_numpy(frame)

            image = np.load(self.images[pos])
            image = torch.from_numpy(image).float()
            fi = image[0:128*52*52].reshape(128, 52, 52)
            loc = image[128*52*52:]

            label = self.labels[pos].split('\t')   # for gt type 2
            if len(label) < 4:
                label = self.labels[pos].split(',') # for gt type 1
            
            # Convert (x1, y1, w, h) into (cx, cy, w, h)
            label = np.array(label, dtype=float)
            label[0] += label[2]/2.
            label[1] += label[3]/2.
            label = torch.as_tensor(label, dtype=torch.float32)
            
            frames.append(frame)
            fis.append(fi)
            locs.append(loc)
            labels.append(label)
        
        return torch.stack(frames, dim=0), torch.stack(fis, dim=0), torch.stack(locs, dim=0), torch.stack(labels, dim=0)

