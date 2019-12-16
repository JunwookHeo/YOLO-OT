# YOT_Base class
from torch.autograd import Variable
from ListContainer import *

class YOT_Base:
    def __init__(self,argvs = []):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = 6
        self.seq_len = 8
        self.img_size = 416
        self.epochs = 20
        
    def run(self):
        self.pre_proc()        
        TotalLoss = []

        for epoch in range(self.epochs):
            totalloss = 0
            listContainer = ListContainer(self.path, self.batch_size, self.seq_len, self.img_size)
            for dataLoader in listContainer:
                pos = 0
                for frames, fis, locs, labels in dataLoader:
                    fis = Variable(fis.to(self.device))
                    locs = Variable(locs.to(self.device))                    
                    labels = Variable(labels.to(self.device), requires_grad=False)

                    totalloss += self.post_proc(epoch, pos, frames, fis, locs, labels)
                    pos += 1

            TotalLoss.append(totalloss)        
            print("Total Loss", TotalLoss)
        print("Model", self.model)
    
    def pre_proc(self):
        pass

    def post_proc(self, epoch, pos, frames, fis, locs, labels):
        for frame, loc, label in zip(frames, locs, labels):
            loc = self.normal_to_locations(frame.shape[0], frame.shape[1], loc)
            print(pos, loc, label)
        
    
    def normal_to_locations(self, wid, ht, locations):
        #print("location in func: ", locations)
        wid *= 1.0
        ht *= 1.0
        locations[0] *= wid
        locations[1] *= ht
        locations[2] *= wid
        locations[3] *= ht
        return locations

    def locations_to_normal(self, wid, ht, locations):
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


    def load_dataset(self):
        pass
    
    def build_model(self):
        pass

