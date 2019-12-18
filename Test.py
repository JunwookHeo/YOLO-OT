import torch
import cv2

from YOT_Base import YOT_Base
from YOTMCLS import *
from YOTMLLP import *
from YOTMCLP import *

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
        self.pm_size = 16

    def processing(self, epoch, pos, frames, fis, locs, locs_mp, labels):
        with torch.no_grad():            
            if(self.pm_size > 0):
                outputs = self.model(fis.float(), locs_mp.float()) 
            else:
                outputs = self.model(fis.float(), locs.float())

            predicts = []
            targets = []
            yolo_targets = []
            img_frames = []
            for i, (frame, output, loc, label) in enumerate(zip(frames, outputs, locs, labels)):
                f = frame[-1]
                o = output[-1]
                l = label[-1]
                y = loc[-1]
                if(self.pm_size > 0):
                    o = coord_utils.probability_map_to_locations(self.pm_size, o)
                p = coord_utils.normal_to_locations(f.size(0), f.size(1), o.clamp(min=0))
                y = coord_utils.normal_to_locations(f.size(0), f.size(1), y.clamp(min=0))
                predicts.append(p)
                targets.append(l)
                yolo_targets.append(y)
                img_frames.append(f)

            self.display_frame(img_frames, yolo_targets, predicts, targets)
                
            iou = self.bbox_iou(torch.stack(predicts, dim=0),  torch.stack(targets, dim=0), False)            
            print("\tIOU : ", iou)
            self.Total_Iou += torch.sum(iou)
    
    def pre_proc(self):
        if(self.pm_size > 0):         
            self.model = YOTMCLS_PM(self.batch_size, self.seq_len).to(self.device)
        else:
            #self.model = YOTMLLP(self.batch_size, self.seq_len).to(self.device)
            #self.model = YOTMCLP(self.batch_size, self.seq_len).to(self.device)
            self.model = YOTMCLS(self.batch_size, self.seq_len).to(self.device)
       

        self.model.load_weights(self.model, self.weights_path)
        self.model.eval()  # Set in evaluation mode

        print(self.model)

    def post_proc(self):
        pass

    def initialize_proc(self, epoch):
        self.Total_Iou = 0

    def finalize_proc(self, epoch):
        print("Total IOU : ", self.Total_Iou)

    def display_frame(self, fs, ys, ps, ts):
        def draw_rectangle(f, p, c, l):
            img = f.numpy()
            c1 = (p[0].int(), p[1].int())
            c2 = (p[0].int() + p[2].int(), p[1].int() + p[3].int())
            cv2.rectangle(img, c1, c2, c, l)
            cv2.imshow("frame", img)
            cv2.waitKey(1)

        for f, y, p, t in zip(fs, ys, ps, ts):
            # Draw rectangle from prediction of YOLO
            draw_rectangle(f, y, (0, 0, 255), 1)
            # Draw rectangle from prediction of YOT
            draw_rectangle(f, p, (0, 255, 0), 1)
            # Draw rectangle from Target
            draw_rectangle(f, t, (255, 0, 0), 1)
        
        

def main(argvs):
    test = Test(argvs)
    test.proc()

if __name__=='__main__':
    main('')
