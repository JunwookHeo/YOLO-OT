import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class VideoDataset(Dataset):
    """ Loading frames in a video file """
    def __init__(self, path, label, seq_num, img_size):
        self.path = path
        self.label = label
        self.seq_num = seq_num
        self.img_size = img_size

        self.cap = cv2.VideoCapture(self.path) 
        assert self.cap.isOpened(), 'Cannot open source'
        
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        pos = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        img = None
        frame = None

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                img = self.resize(frame, self.img_size)
                
            break

        return frame, img

    def resize(self, frame, img_size):
        s = frame.shape[:2]
        r = img_size/max(s)
        ns = tuple([int(s[1]*r), int(s[0]*r)])
        img = cv2.resize(frame, ns, interpolation = cv2.INTER_CUBIC)
        
        d_w = img_size - ns[0]
        d_h = img_size - ns[1]
        top, bottom = d_h//2, d_h - d_h//2
        left, right = d_w//2, d_w - d_w//2

        color = [128, 128, 128]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        assert(img_size == img.shape[0])
        assert(img_size == img.shape[1])

        img = img[:,:,::-1].transpose((2,0,1)).copy()
        img = torch.from_numpy(img).float().div(255.0)

        return img

    

