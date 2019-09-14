import cv2
import numpy as np
import os.path
import time
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from YOTM import *
from ListContainer import *

import os,sys
yolodir = os.path.join(os.path.dirname(__file__), 'YOLOv3')
sys.path.append(yolodir)
print(yolodir)

from YOLOv3.models import *


class YOLO_OT:
    def __init__(self,argvs = []):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = 1
        self.input_size = 1024
        self.hidden_size = 256
        self.output_size = 6
        self.num_sequence = 6
        self.num_layers =3
        self.img_size = 416
       
        self.yolo = Darknet('YOLOv3/config/yolov3.cfg', img_size=self.img_size).to(self.device)
        weights_path = 'YOLOv3/weights/yolov3.weights'

        if weights_path.endswith(".weights"):
            # Load darknet weights
            self.yolo.load_darknet_weights(weights_path)
        else:
            # Load checkpoint weights
            self.yolo.load_state_dict(torch.load(weights_path))


        self.yotm = YOTM(self.batch_size, self.input_size, self.hidden_size, self.output_size, self.num_sequence, self.num_layers).to(self.device)
    
    def train(self):
        self.yolo.eval()

        for epoch in range(1):
            listContainer = ListContainer('Dataset/Sample', self.batch_size, self.num_sequence, self.img_size)
            for dataLoader in listContainer:
                print(dataLoader)
                for frames, imgs in dataLoader:
                    for img in imgs:
                        if img is not None:
                            img = Variable(img.to(self.device))
                            
                            with torch.no_grad():
                                output = self.yolo(img)
                            output = non_max_suppression(output, 0.8, 0.4)
                            #print(type(img))
                            #self.model(img)
                            #cv2.imshow('frame', np.array(img))
                            #cv2.waitKey(1)
                            pass
        
def main(argvs):
    ot = YOLO_OT(argvs)
    ot.train()

if __name__=='__main__':
    main('')
