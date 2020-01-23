import torch

import pandas as pd
import argparse
import importlib
import time

from YOT_Base import YOT_Base
from ListContainer import *

from YOTMLLP import *
from YOTMCLP import *
from YOTMCLS import *
from YOTMONEL import *
from YOTMROLO import *

from YOTMCLS_PM import *
from YOTMLLP_PM import *

from YOTMCLSM import*

from coord_utils import *
from logger import logger as LOG

class Train(YOT_Base):
    def __init__(self,argvs = []):
        super(Train, self).__init__(argvs)
        
        self.log_interval = 1
        self.result_columns=['Train_Loss', 'Validate_Loss', 'Train_YOT_IoU', 'Validate_YOT_IoU', 'Train_YOLO_IoU', 'Validate_YOLO_IoU']
        self.Report = pd.DataFrame(columns=self.result_columns)

        ## Change configuration
        opt = self.update_config()

        self.data_path = opt.data_path
        self.epochs = opt.epochs
        self.save_weights = opt.save_weights
        self.mode = opt.run_mode
        self.model_name = opt.model_name

    def update_config(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--data_path", type=str, default="../rolo_data", help="path to data config file")
        parser.add_argument("--epochs", type=int, default=30, help="size of epoch")
        parser.add_argument("--save_weights", type=bool, default=True, help="save checkpoint and weights")
        parser.add_argument("--run_mode", type=str, default="train", help="train or test mode")
        parser.add_argument("--model_name", type=str, default="YOTMLLP", help="class name of the model")

        args, _ = parser.parse_known_args()
        return args

    def processing(self, epoch, lpos, dpos, frames, fis, locs, labels):
        outputs = self.model(fis, locs)
        
        img_frames = self.get_last_sequence(frames)
        predicts = self.get_last_sequence(outputs)
        yolo_locs = self.get_last_sequence(locs) 
        targets = self.get_last_sequence(labels)

        for i, (f, l) in enumerate(zip(img_frames, targets)):
            targets[i] = coord_utils.location_to_normal(f.shape[1], f.shape[0], l)
        
        target_values = self.model.get_targets(targets.clone())
        loss = self.loss(predicts, target_values)
        #loss = self.model.get_weighted_loss(predicts, target_values, torch.tensor([2,2,1,1], dtype=torch.float32))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.sum_loss += float(loss.data)
        self.frame_cnt += len(predicts)
        
        if dpos % self.log_interval == 0:
            LOG.info('Train pos: {}-{}-{} [Loss: {:.6f}]'.format(epoch, lpos, dpos, loss.data/len(predicts)))
            predict_boxes = []
            yolo_boxes = []
            target_boxes = []        
            for i, (f, p, y, t) in enumerate(zip(img_frames, predicts, yolo_locs, targets)):
                p = self.model.get_location(p)
                predict_boxes.append(coord_utils.normal_to_location(f.shape[1], f.shape[0], p))
                yolo_boxes.append(coord_utils.normal_to_location(f.shape[1], f.shape[0], y))
                target_boxes.append(coord_utils.normal_to_location(f.shape[1], f.shape[0], t))
                LOG.debug(f"\tPredict:{p.int().data.numpy()},    YOLO:{y[0:4].int().data.numpy()},    GT:{t.int().data.numpy()}")
            iou = coord_utils.bbox_iou(torch.stack(predict_boxes, dim=0),  torch.stack(target_boxes, dim=0), False)
            yiou = coord_utils.bbox_iou(torch.stack(yolo_boxes, dim=0),  torch.stack(target_boxes, dim=0), False)
            self.sum_iou[0] += float(torch.sum(iou))
            self.sum_iou[1] += float(torch.sum(yiou))
                        
            LOG.info(f"\tIOU : {iou.data}")
    
    def pre_proc(self):
        m = importlib.import_module(self.model_name)
        mobj = getattr(m, self.model_name)
        self.model = mobj(self.batch_size, self.seq_len).to(self.device)
        LOG.info(f'\n{self.model}')

        self.loss = self.model.get_loss_function()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        self.model.load_checkpoint(self.model, self.optimizer, self.weights_path)
        self.model.train()  # Set in training mode
        
    def post_proc(self):
        LOG.info(f'\n{self.model}')
        if self.save_weights is True:
            self.model.save_weights(self.model, self.weights_path)

    def initialize_list_loop(self, name):
        self.list_name = name

    def finalize_list_loop(self):
        pass

    def initialize_epoch_processing(self, epoch):
        self.sum_loss = 0
        self.sum_iou = [0, 0]
        self.frame_cnt = 0

    def finalize_epoch_processing(self, epoch):
        avg_loss = self.sum_loss/self.frame_cnt
        train_iou = [self.sum_iou[0]/self.frame_cnt, self.sum_iou[1]/self.frame_cnt]
        validate_loss, validate_iou = self.evaluation()        
        self.model.train()
        
        if self.save_weights is True:
            self.model.save_checkpoint(self.model, self.optimizer, self.weights_path)

        self.Report = self.Report.append({self.result_columns[0]:avg_loss, self.result_columns[1]:validate_loss, 
                                        self.result_columns[2]:train_iou[0], self.result_columns[3]:validate_iou[0], 
                                        self.result_columns[4]:train_iou[1], self.result_columns[5]:validate_iou[1]}, ignore_index=True)
        LOG.info(f'\n{self.Report}')

    def evaluation(self):
        total_iou = [0, 0]
        total_cnt = 0
        loss = 0
        self.model.train(False)

        eval_list = ListContainer(self.data_path, self.batch_size, self.seq_len, self.img_size, 'test')
        for dataLoader in eval_list:
            for frames, fis, locs, labels in dataLoader:
                fis = Variable(fis.to(self.device))
                locs = Variable(locs.to(self.device))
                labels = Variable(labels.to(self.device), requires_grad=False)

                with torch.no_grad():            
                    outputs = self.model(fis, locs)
        
                    img_frames = self.get_last_sequence(frames)
                    predicts = self.get_last_sequence(outputs)                
                    yolo_predicts = self.get_last_sequence(locs)
                    targets = self.get_last_sequence(labels)

                    norm_targets = targets.clone()
                    for i, (f, l) in enumerate(zip(img_frames, norm_targets)):
                        norm_targets[i] = coord_utils.location_to_normal(f.shape[1], f.shape[0], l)
                    
                    target_values = self.model.get_targets(norm_targets)
                    loss += self.loss(predicts, target_values)

                    predict_boxes = []
                    for i, (f, o, y) in enumerate(zip(img_frames, predicts, yolo_predicts)):
                        o = self.model.get_location(o)
                        predict_boxes.append(coord_utils.normal_to_location(f.size(1), f.size(0), o.clamp(min=0)))
                        yolo_predicts[i] = coord_utils.normal_to_location(f.size(1), f.size(0), y.clamp(min=0))
                        
                    iou = coord_utils.bbox_iou(torch.stack(predict_boxes, dim=0),  targets, False)
                    yiou = coord_utils.bbox_iou(yolo_predicts.float(),  targets, False)
                    total_iou[0] += float(torch.sum(iou))
                    total_iou[1] += float(torch.sum(yiou))
                    total_cnt += len(iou)
        
        return float(loss/total_cnt), [float(total_iou[0]/total_cnt), float(total_iou[1]/total_cnt)]


def main(argvs):
    train = Train(argvs)
    train.proc()

if __name__=='__main__':
    main('')

