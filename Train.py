import torch

import pandas as pd
import argparse
import importlib

from YOT_Base import YOT_Base
from ListContainer import *

from YOTMLLP import *
from YOTMCLP import *
from YOTMCLS import *
from YOTMONEL import *
from YOTMROLO import *

from YOTMCLS_PM import *
from YOTMLLP_PM import *

from coord_utils import *

class Train(YOT_Base):
    def __init__(self,argvs = []):
        super(Train, self).__init__(argvs)
        
        self.log_interval = 1        
        self.Report = pd.DataFrame(columns=['Loss', 'Train IoU', 'Validate IoU'])

        ## Change configuration
        opt = self.update_config()

        self.epochs = opt.epochs
        self.save_weights = opt.save_weights
        self.mode = opt.run_mode
        self.model_name = opt.model_name

    def update_config(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--epochs", type=int, default=10, help="size of epoch")
        parser.add_argument("--save_weights", type=bool, default=True, help="save checkpoint and weights")
        parser.add_argument("--run_mode", type=str, default="train", help="train or test mode")
        parser.add_argument("--model_name", type=str, default="YOTMLLP", help="class name of the model")

        return parser.parse_args()

    def processing(self, epoch, pos, frames, fis, locs, labels):
        outputs = self.model(fis, locs)
        
        img_frames = self.get_last_sequence(frames)
        predicts = self.get_last_sequence(outputs)                
        targets = self.get_last_sequence(labels)

        for i, (f, l) in enumerate(zip(img_frames, targets)):
            targets[i] = coord_utils.location_to_normal(f.shape[0], f.shape[1], l)
        
        target_values = self.model.get_targets(targets)
        loss = self.loss(predicts, target_values)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.sum_loss += float(loss.data)
        self.frame_cnt += len(predicts)
        
        if pos % self.log_interval == 0:
            print('Train pos: {}-{} [Loss: {:.6f}]'.format(epoch, pos, loss.data/len(predicts)))
            predict_boxes = []
            target_boxes = []
            for i, (f, p, t) in enumerate(zip(img_frames, predicts, targets)):
                p = self.model.get_location(p)
                predict_boxes.append(coord_utils.normal_to_location(f.shape[0], f.shape[1], p))
                target_boxes.append(coord_utils.normal_to_location(f.shape[0], f.shape[1], t))
                print("\t", p, t)
            iou = coord_utils.bbox_iou(torch.stack(predict_boxes, dim=0),  torch.stack(target_boxes, dim=0), False)
            self.sum_iou += float(torch.sum(iou))
                        
            print("\tIOU : ", iou)
    
    def iou_loss(self, frms, predicts, targets):
        for f, p, t in zip(frms, predicts, targets):
            self.normal_to_location(f.shape[1], f.shape[2], p)
            self.normal_to_location(f.shape[1], f.shape[2], t)

        iou = self.bbox_iou(predicts, targets, False)
        return torch.sum(1-iou)

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

        print(self.model)
        self.loss = self.model.get_loss_function() # nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        self.model.load_checkpoint(self.model, self.optimizer, self.weights_path)
        self.model.train()  # Set in training mode
        
    def post_proc(self):
        print("Model", self.model)
        if self.save_weights is True:
            self.model.save_weights(self.model, self.weights_path)

    def initialize_proc(self, epoch):
        self.sum_loss = 0
        self.sum_iou = 0
        self.frame_cnt = 0

    def finalize_proc(self, epoch):
        avg_loss = self.sum_loss/self.frame_cnt
        train_iou = self.sum_iou/self.frame_cnt        
        validate_iou = self.evaluation()        
        self.model.train()
        
        if self.save_weights is True:
            self.model.save_checkpoint(self.model, self.optimizer, self.weights_path)

        self.Report = self.Report.append({'Loss':avg_loss, 'Train IoU':train_iou, 'Validate IoU':validate_iou}, ignore_index=True)
        print(self.Report)

    def evaluation(self):
        total_iou = 0
        total_cnt = 0
        self.model.train(False)

        eval_list = ListContainer(self.data_path, self.batch_size, self.seq_len, self.img_size, 'test')
        for dataLoader in eval_list:
            for frames, fis, locs, labels in dataLoader:
                fis = Variable(fis.to(self.device))
                locs = Variable(locs.to(self.device))
                labels = Variable(labels.to(self.device), requires_grad=False)

                with torch.no_grad():            
                    outputs = self.model(fis.float(), locs.float())
        
                    img_frames = self.get_last_sequence(frames)
                    predicts = self.get_last_sequence(outputs)                
                    yolo_predicts = self.get_last_sequence(locs)
                    targets = self.get_last_sequence(labels)

                    predict_boxes = []
                    for i, (f, o, y, l) in enumerate(zip(img_frames, predicts, yolo_predicts, targets)):
                        o = self.model.get_location(o)
                        predict_boxes.append(coord_utils.normal_to_location(f.size(0), f.size(1), o.clamp(min=0)))
                        yolo_predicts[i] = coord_utils.normal_to_location(f.size(0), f.size(1), y.clamp(min=0))
                        
                    iou = coord_utils.bbox_iou(torch.stack(predict_boxes, dim=0),  targets, False)
                    yiou = coord_utils.bbox_iou(yolo_predicts.float(),  targets, False)
                    #print(f"\tIOU : {iou} - {yiou}")
                    total_iou += float(torch.sum(iou))
                    total_cnt += len(iou)
        
        return float(total_iou/total_cnt)


def main(argvs):
    train = Train(argvs)
    train.proc()

if __name__=='__main__':
    main('')
