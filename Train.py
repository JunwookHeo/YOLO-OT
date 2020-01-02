import torch
import torch.nn.functional as F

from YOT_Base import YOT_Base
from YOTMLLP import *
from YOTMCLP import *
from YOTMCLS import *
from YOTMONEL import *
from YOTMROLO import *

from YOTMCLS_PM import *

from coord_utils import *

class Train(YOT_Base):
    def __init__(self,argvs = []):
        super(Train, self).__init__(argvs)
        self.log_interval = 1
        # The path of dataset
        self.path = "../rolo_data"         
        
        self.TotalLoss = []
        self.frame_cnt = 0
        self.epochs = 6
        self.pm_size = CLSMPMParam.LocMapSize

    def processing(self, epoch, pos, frames, fis, locs, locs_pm, labels):
        if(self.pm_size > 0):
            outputs = self.model(fis.float(), locs_pm.float()) 
        else:
            outputs = self.model(fis.float(), locs.float())
        
        img_frames = self.get_last_sequence(frames)
        predicts = self.get_last_sequence(outputs)                
        targets = self.get_last_sequence(labels)

        targets_pm = []
        for i, (f, l) in enumerate(zip(img_frames, targets)):
            targets[i] = coord_utils.locations_to_normal(f.shape[0], f.shape[1], l)
            if(self.pm_size > 0):
                pm = coord_utils.locations_to_probability_map(self.pm_size, l)
                pm = pm.view(-1)
                targets_pm.append(pm)
        
        if(self.pm_size > 0):
            loss = self.loss(predicts, torch.stack(targets_pm, dim=0))            
        else:
            loss = self.loss(predicts, targets)
            #loss = self.iou_loss(img_frames, predicts, targets)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.sum_loss += loss.data
        self.frame_cnt += len(predicts)
        
        if pos % self.log_interval == 0:
            print('Train pos: {}-{} [Loss: {:.6f}]'.format(epoch, pos, loss.data/len(predicts)))
            predict_boxes = []
            target_boxes = []
            for i, (f, p, t) in enumerate(zip(img_frames, predicts, targets)):
                if(self.pm_size > 0):
                    p = coord_utils.probability_map_to_locations(self.pm_size, p)
                predict_boxes.append(coord_utils.normal_to_locations(f.shape[0], f.shape[1], p))
                target_boxes.append(coord_utils.normal_to_locations(f.shape[0], f.shape[1], t))
                print("\t", p, t)
            iou = coord_utils.bbox_iou(torch.stack(predict_boxes, dim=0),  torch.stack(target_boxes, dim=0), False)
            print("\tIOU : ", iou)
                        
    
    def iou_loss(self, frms, predicts, targets):
        for f, p, t in zip(frms, predicts, targets):
            self.normal_to_locations(f.shape[1], f.shape[2], p)
            self.normal_to_locations(f.shape[1], f.shape[2], t)

        iou = self.bbox_iou(predicts, targets, False)
        return torch.sum(1-iou)

    def pre_proc(self):        
        if(self.pm_size > 0):         
            self.model = YOTMCLS_PM(self.batch_size, self.seq_len).to(self.device)
        else:
            #self.model = YOTMLLP(self.batch_size, self.seq_len).to(self.device)
            #self.model = YOTMCLP(self.batch_size, self.seq_len).to(self.device)
            self.model = YOTMCLS(self.batch_size, self.seq_len).to(self.device)
            #self.model = YOTMONEL(self.batch_size, self.seq_len).to(self.device)
            #self.model = YOTMROLO(self.batch_size, self.seq_len).to(self.device)

        print(self.model)
        self.loss = nn.MSELoss(reduction='sum')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)

        self.model.load_checkpoint(self.model, self.optimizer, self.check_path)

        self.model.train()  # Set in training mode
        
    def post_proc(self):
        print("Model", self.model)
        self.model.save_weights(self.model, self.weights_path)

    def initialize_proc(self, epoch):
        self.sum_loss = 0
        self.frame_cnt = 0

    def finalize_proc(self, epoch):
        self.TotalLoss.append(self.sum_loss/self.frame_cnt)        
        print("Avg Loss", self.TotalLoss)

        self.model.save_checkpoint(self.model, self.optimizer, self.check_path)

def main(argvs):
    train = Train(argvs)
    train.proc()

if __name__=='__main__':
    main('')
