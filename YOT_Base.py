# YOT_Base class

from abc import ABC, abstractmethod
from torch.autograd import Variable
from ListContainer import *

import argparse

class YOT_Base(ABC):
    def __init__(self,argvs = []):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        opt = self.parse_config()

        self.data_path = opt.data_config
        self.batch_size = opt.batch_size
        self.seq_len = opt.sequence_length
        self.img_size = opt.img_size
        self.epochs = opt.epochs
        
        self.weights_path = opt.weights_path
        self.save_weights = opt.save_weights

        self.mode = opt.run_mode
        self.model_name = opt.model_name

    def parse_config(self):
        parser = argparse.ArgumentParser()

        # default argument
        parser.add_argument("--data_config", type=str, default="../rolo_data", help="path to data config file")
    
        parser.add_argument("--epochs", type=int, default=30, help="size of epoch")
        parser.add_argument("--batch_size", type=int, default=6, help="size of each image batch")
        parser.add_argument("--sequence_length", type=int, default=6, help="size of each sequence of LSTM")
        parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
        
        parser.add_argument("--weights_path", type=str, default="outputs/weights", help="path to weights folder")
        parser.add_argument("--save_weights", type=bool, default=False, help="save checkpoint and weights")

        parser.add_argument("--run_mode", type=str, default="none", help="train or test mode")        
        parser.add_argument("--model_name", type=str, default="YOTMLLP", help="class name of the model")

        return parser.parse_args()

    def proc(self):
        self.pre_proc()

        for epoch in range(self.epochs):            
            self.initialize_proc(epoch)
            listContainer = ListContainer(self.data_path, self.batch_size, self.seq_len, self.img_size, self.mode)
            for dataLoader in listContainer:
                pos = 0
                for frames, fis, locs, labels in dataLoader:
                    fis = Variable(fis.to(self.device))
                    locs = Variable(locs.to(self.device))
                    labels = Variable(labels.to(self.device), requires_grad=False)

                    self.processing(epoch, pos, frames, fis, locs, labels)
                    pos += 1

            self.finalize_proc(epoch)
        
        self.post_proc()

    @abstractmethod
    def pre_proc(self):
        pass

    @abstractmethod
    def processing(self, epoch, pos, frames, fis, locs, labels):
        pass

    @abstractmethod
    def post_proc(self):
        pass
        
    @abstractmethod
    def initialize_proc(self, epoch):
        pass

    @abstractmethod
    def finalize_proc(self, epoch):
        pass
    
    def get_last_sequence(self, data):
        d = torch.split(data, self.seq_len -1, dim=1)
        return torch.squeeze(d[1], dim=1)

    def normal_to_location(self, wid, ht, locations):
        #print("location in func: ", locations)
        wid *= 1.0
        ht *= 1.0
        locations[0] *= wid
        locations[1] *= ht
        locations[2] *= wid
        locations[3] *= ht
        return locations
    '''
    def location_to_normal(self, wid, ht, locations):
        #print("location in func: ", locations)
        wid *= 1.0
        ht *= 1.0
        locations[0] /= wid
        locations[1] /= ht
        locations[2] /= wid
        locations[3] /= ht
        return locations

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        Returns the IoU of two bounding boxes
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0
        )
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou
    '''

