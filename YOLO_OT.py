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
from YOLOv3.utils.utils import *
import matplotlib.pyplot as plt



class YOLO_OT:
    def __init__(self,argvs = []):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = 6
        self.input_size = 1024
        self.hidden_size = 256
        self.output_size = 6
        self.num_sequence = 6
        self.num_layers =3
        self.img_size = 416

        self.classes = load_classes("YOLOv3/data/coco.names")       
        self.yolo = Darknet('YOLOv3/config/yolov3.cfg', img_size=self.img_size).to(self.device)
        weights_path = 'YOLOv3/weights/yolov3.weights'
        if weights_path.endswith(".weights"):
            self.yolo.load_darknet_weights(weights_path)
        else:
            self.yolo.load_state_dict(torch.load(weights_path))

        self.yotm = YOTM(self.batch_size, self.input_size, self.hidden_size, self.output_size, self.num_sequence, self.num_layers).to(self.device)
    
    def train(self):
        self.yolo.eval()

        for epoch in range(1):
            listContainer = ListContainer('Dataset/Sample', self.batch_size, self.num_sequence, self.img_size)
            for dataLoader in listContainer:
                print(dataLoader)
                for frames, imgs in dataLoader:
                    imgs = Variable(imgs.to(self.device))
                    
                    with torch.no_grad():
                        outputs = self.yolo(imgs)
                    outputs = non_max_suppression(outputs, 0.8, 0.4)

                    self.post_proc(frames, outputs)

    def post_proc(self, frames, detections):
        cmap = np.random.randint(0, 256, size=(32, 3))
        colors = [c for c in cmap]        

        for i, (img, detection) in enumerate(zip(frames, detections)):
            img = img.permute(0, 1, 2).numpy()
            if detection is not None:
                detection = rescale_boxes(detection, self.img_size, img.shape[:2])
                unique_labels = detection[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    print("\t+ Label: %s, Conf: %.5f" % (self.classes[int(cls_pred)], cls_conf.item()))
                    c1 = (x1.int(), y1.int())
                    c2 = (x2.int(), y2.int())
                    c = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    color = (int(c[0]), int(c[1]), int(c[2]))
                    cv2.rectangle(img, c1, c2, color, 1)
                    t_size = cv2.getTextSize(self.classes[int(cls_pred)], cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
                    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                    cv2.rectangle(img, c1, c2,color, -1)
                    cv2.putText(img, self.classes[int(cls_pred)], (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);

                cv2.imshow("frame", img)
                key = cv2.waitKey(1)


def main(argvs):
    ot = YOLO_OT(argvs)
    ot.train()

if __name__=='__main__':
    main('')
