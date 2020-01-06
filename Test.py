import torch
import cv2

from YOT_Base import YOT_Base
from ListContainer import *

from YOTMCLS import *
from YOTMLLP import *
from YOTMCLP import *
from YOTMONEL import *
from YOTMROLO import *

from YOTMCLS_PM import *

from coord_utils import *

class Test(YOT_Base):
    def __init__(self,argvs = []):
        super(Test, self).__init__(argvs)
        # The path of dataset
        self.path = "../rolo_data" 

        self.epochs = 1
        self.batch_size = 1
        self.Total_Iou = 0
        self.Total_cnt = 0
        self.pm_size = 0 #CLSMPMParam.LocMapSize

    def proc(self):
        self.pre_proc()

        for epoch in range(self.epochs):            
            self.initialize_proc(epoch)
            listContainer = ListContainer(self.path, self.batch_size, self.seq_len, self.pm_size, 'test')
            for dataLoader in listContainer:
                pos = 0
                for frames, fis, locs, locs_mp, labels in dataLoader:
                    fis = Variable(fis.to(self.device))
                    locs = Variable(locs.to(self.device))
                    locs_mp = Variable(locs_mp.to(self.device)) 
                    labels = Variable(labels.to(self.device), requires_grad=False)

                    self.processing(epoch, pos, frames, fis, locs, locs_mp, labels)
                    pos += 1

            self.finalize_proc(epoch)
        
        self.post_proc()

    def processing(self, epoch, pos, frames, fis, locs, locs_mp, labels):
        with torch.no_grad():            
            if(self.pm_size > 0):
                outputs = self.model(fis.float(), locs_mp.float()) 
            else:
                outputs = self.model(fis.float(), locs.float())
            
            img_frames = self.get_last_sequence(frames)
            predicts = self.get_last_sequence(outputs)                
            yolo_predicts = self.get_last_sequence(locs)
            targets = self.get_last_sequence(labels)

            predict_boxes = []
            for i, (f, o, y, l) in enumerate(zip(img_frames, predicts, yolo_predicts, targets)):
                if(self.pm_size > 0):
                    o = coord_utils.probability_map_to_locations(self.pm_size, o)
                predict_boxes.append(coord_utils.normal_to_locations(f.size(0), f.size(1), o.clamp(min=0)))
                yolo_predicts[i] = coord_utils.normal_to_locations(f.size(0), f.size(1), y.clamp(min=0))

            self.display_frame(img_frames, yolo_predicts, torch.stack(predict_boxes, dim=0), targets)
                
            iou = self.bbox_iou(torch.stack(predict_boxes, dim=0),  targets, False)
            yiou = self.bbox_iou(yolo_predicts.float(),  targets, False)
            print(f"\t{pos} IOU : {iou} - {yiou}")
            self.Total_Iou += torch.sum(iou)
            self.Total_cnt += len(iou) 

    def pre_proc(self):
        if(self.pm_size > 0):         
            self.model = YOTMCLS_PM(self.batch_size, self.seq_len).to(self.device)
        else:
            #self.model = YOTMLLP(self.batch_size, self.seq_len).to(self.device)
            #self.model = YOTMCLP(self.batch_size, self.seq_len).to(self.device)
            self.model = YOTMCLS(self.batch_size, self.seq_len).to(self.device)
            #self.model = YOTMONEL(self.batch_size, self.seq_len).to(self.device)
            #self.model = YOTMROLO(self.batch_size, self.seq_len).to(self.device)

        self.model.load_weights(self.model, self.weights_path)
        self.model.eval()  # Set in evaluation mode

        print(self.model)

    def post_proc(self):
        pass

    def initialize_proc(self, epoch):
        self.Total_Iou = 0
        self.Total_cnt = 0

    def finalize_proc(self, epoch):
        print("Avg IOU : ", self.Total_Iou/self.Total_cnt)

    def display_frame(self, fs, ys, ps, ts):
        def draw_rectangle(img, p, c, l):            
            c1 = (p[0].int(), p[1].int())
            c2 = (p[0].int() + p[2].int(), p[1].int() + p[3].int())
            cv2.rectangle(img, c1, c2, c, l)

        for f, y, p, t in zip(fs, ys, ps, ts):
            img = f.numpy()
            # Draw rectangle from prediction of YOLO
            draw_rectangle(img, y, (255, 0, 0), 2)
            # Draw rectangle from prediction of YOT
            draw_rectangle(img, p, (0, 255, 0), 2)
            # Draw rectangle from Target
            draw_rectangle(img, t, (0, 0, 255), 1)

            cv2.imshow("frame", img)
            cv2.waitKey(1)
        
        

def main(argvs):
    test = Test(argvs)
    test.proc()

if __name__=='__main__':
    main('')
