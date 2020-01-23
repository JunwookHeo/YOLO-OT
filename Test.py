import torch
import cv2
import argparse
import importlib
import os, time

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
        
        self.save_coord_list = False
        ## Change configuration
        opt = self.update_config()

        self.data_path = opt.data_path
        self.epochs = opt.epochs
        self.batch_size = opt.batch_size
        self.mode = opt.run_mode
        self.model_name = opt.model_name

    def update_config(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--data_path", type=str, default="../rolo_data", help="path to data config file")
        parser.add_argument("--epochs", type=int, default=1, help="size of epoch")
        parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
        parser.add_argument("--run_mode", type=str, default="test", help="train or test mode")
        parser.add_argument("--model_name", type=str, default="YOTMLLP", help="class name of the model")
        
        args, _ = parser.parse_known_args()
        return args


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
                predict_boxes.append(coord_utils.normal_to_location(f.size(1), f.size(0), o.clamp(min=0)))
                yolo_predicts[i] = coord_utils.normal_to_location(f.size(1), f.size(0), y.clamp(min=0))

            self.display_frame(img_frames, torch.stack(predict_boxes, dim=0), yolo_predicts, targets)
            self.append_coord_log(torch.stack(predict_boxes, dim=0), yolo_predicts, targets)

            iou = coord_utils.bbox_iou(torch.stack(predict_boxes, dim=0),  targets, False)
            yiou = coord_utils.bbox_iou(yolo_predicts.float(),  targets, False)
            LOG.debug(f"\t{lpos}-{dpos} IOU : {iou} - {yiou}")
            self.Total_Iou[0] += float(torch.sum(iou))
            self.Total_Iou[1] += float(torch.sum(yiou))
            self.Total_cnt += len(iou) 

    def pre_proc(self):
        m = importlib.import_module(self.model_name)
        mobj = getattr(m, self.model_name)
        self.model = mobj(self.batch_size, self.seq_len).to(self.device)

        self.model.load_weights(self.model, self.weights_path)
        self.model.eval()  # Set in evaluation mode

        self.strtime = time.strftime("%Y%m%d_%H%M%S")

    def post_proc(self):
        LOG.info(f'\n{self.model}')

    def initialize_list_loop(self, name):
        self.list_name = name
        if self.save_coord_list:
            self.list_log = np.empty((0, 3, 4), int)

    def finalize_list_loop(self):
        if self.save_coord_list:
            path = os.path.join('./outputs', 'coord_' + self.model_name + '_' + self.strtime)
            if not os.path.exists(path):
                os.makedirs(path)
            name = os.path.join(path, self.list_name)
            np.save(name, self.list_log)

    def initialize_epoch_processing(self, epoch):
        self.Total_Iou = [0., 0.]
        self.Total_cnt = 0

    def finalize_epoch_processing(self, epoch):
        LOG.info("Avg IOU : YOT={:f}, YOLO={:f}".format(self.Total_Iou[0]/self.Total_cnt, self.Total_Iou[1]/self.Total_cnt))

    def display_frame(self, fs, ps, ys, ts):
        def draw_rectangle(img, p, c, l):            
            c1 = ((p[0] - p[2]/2.).int(), (p[1] - p[3]/2.).int())
            c2 = ((p[0] + p[2]/2.).int(), (p[1] + p[3]/2.).int())
            cv2.rectangle(img, c1, c2, c, l)

        for f, p, y, t in zip(fs, ps, ys, ts):
            img = f.numpy()
            # Draw rectangle from prediction of YOLO
            draw_rectangle(img, y, (255, 0, 0), 2)
            # Draw rectangle from prediction of YOT
            draw_rectangle(img, p, (0, 255, 0), 2)
            # Draw rectangle from Target
            draw_rectangle(img, t, (0, 0, 255), 1)

            cv2.imshow("frame", img)
            cv2.waitKey(1)
        
    def append_coord_log(self, ps, ys, ts):
        if self.save_coord_list:
            for p, y, t in zip(ps, ys[..., 0:4], ts):
                self.list_log = np.append(self.list_log, np.array([[p.numpy(), y.numpy(), t.numpy()]]), axis=0)

def main(argvs):
    test = Test(argvs)
    test.proc()

if __name__=='__main__':
    main('')
