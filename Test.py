import torch
import cv2
import argparse
import importlib

from YOT_Base import YOT_Base
from ListContainer import *

from YOTMCLS import *
from YOTMLLP import *
from YOTMCLP import *
from YOTMONEL import *
from YOTMROLO import *

from YOTMCLS_PM import *
from YOTMLLP_PM import *

from coord_utils import *
from logger import logger as LOG

class Test(YOT_Base):
    def __init__(self,argvs = []):
        super(Test, self).__init__(argvs)
        ## Change configuration
        opt = self.update_config()

        self.epochs = opt.epochs
        self.batch_size = opt.batch_size
        self.mode = opt.run_mode
        self.model_name = opt.model_name

    def update_config(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--epochs", type=int, default=1, help="size of epoch")
        parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
        parser.add_argument("--run_mode", type=str, default="test", help="train or test mode")
        parser.add_argument("--model_name", type=str, default="YOTMLLP", help="class name of the model")
        
        return parser.parse_args()

    def processing(self, epoch, lpos, dpos, frames, fis, locs, labels):
        with torch.no_grad():            
            outputs = self.model(fis, locs)
            
            img_frames = self.get_last_sequence(frames)
            predicts = self.get_last_sequence(outputs)                
            yolo_predicts = self.get_last_sequence(locs)
            targets = self.get_last_sequence(labels)

            predict_boxes = []
            for i, (f, o, y, l) in enumerate(zip(img_frames, predicts, yolo_predicts, targets)):
                o = self.model.get_location(o)
                predict_boxes.append(coord_utils.normal_to_location(f.size(0), f.size(1), o.clamp(min=0)))
                yolo_predicts[i] = coord_utils.normal_to_location(f.size(0), f.size(1), y.clamp(min=0))

            self.display_frame(img_frames, yolo_predicts, torch.stack(predict_boxes, dim=0), targets)
                
            iou = coord_utils.bbox_iou(torch.stack(predict_boxes, dim=0),  targets, False)
            yiou = coord_utils.bbox_iou(yolo_predicts.float(),  targets, False)
            LOG.debug(f"\t{lpos}-{dpos} IOU : {iou} - {yiou}")
            self.Total_Iou += float(torch.sum(iou))
            self.Total_cnt += len(iou) 

    def pre_proc(self):
        m = importlib.import_module(self.model_name)
        mobj = getattr(m, self.model_name)
        self.model = mobj(self.batch_size, self.seq_len).to(self.device)

        ### Models Using Probability Map
        #self.model = YOTMLLP_PM(self.batch_size, self.seq_len).to(self.device)
        #self.model = YOTMCLS_PM(self.batch_size, self.seq_len).to(self.device)
        
        ### Models Without Probability Map
        #self.model = YOTMLLP(self.batch_size, self.seq_len).to(self.device)
        #self.model = YOTMCLP(self.batch_size, self.seq_len).to(self.device)
        #self.model = YOTMCLS(self.batch_size, self.seq_len).to(self.device)
        #self.model = YOTMONEL(self.batch_size, self.seq_len).to(self.device)
        #self.model = YOTMROLO(self.batch_size, self.seq_len).to(self.device)

        self.model.load_weights(self.model, self.weights_path)
        self.model.eval()  # Set in evaluation mode

    def post_proc(self):
        LOG.info(f'{self.model}')

    def initialize_processing(self, epoch):
        self.Total_Iou = 0
        self.Total_cnt = 0

    def finalize_processing(self, epoch):
        LOG.info("Avg IOU : {:f}".format(self.Total_Iou/self.Total_cnt))

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
