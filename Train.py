import torch
import torch.nn.functional as F

from YOT_Base import YOT_Base
from YOTMLLP import *
from YOTMCLP import *
from YOTMCLS import *

class Train(YOT_Base):
    def __init__(self,argvs = []):
        super(Train, self).__init__(argvs)
        self.log_interval = 1
        # The path of dataset
        self.path = "../rolo_data"         
        
        self.TotalLoss = []
        self.epochs = 2

    def processing(self, epoch, pos, frames, fis, locs, labels):
        outputs = self.model(fis.float(), locs.float())

        predicts = []
        targets = []
        frms = []
        for i, (f, output, label) in enumerate(zip(frames, outputs, labels)):
            p = output[-1]
            l = label[-1]
            l = self.locations_to_normal(f.shape[1], f.shape[2], l)
            predicts.append(p)
            targets.append(l)
            frms.append(f)

        loss = self.loss(torch.stack(predicts, dim=0), torch.stack(targets, dim=0))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        if pos % self.log_interval == 0:
            print('Train pos: {}-{} [Loss: {:.6f}]'.format(epoch, pos, loss.data))

            for f, p, t in zip(frms, predicts, targets):
                print("\t", 
                        self.normal_to_locations(f.shape[1], f.shape[2], p),
                        self.normal_to_locations(f.shape[1], f.shape[2], t)
                        )
            iou = self.bbox_iou(torch.stack(predicts, dim=0),  torch.stack(targets, dim=0), False)            
            print("\tIOU : ", iou)
        
        self.sum_loss += loss.data
    
    def pre_proc(self):        
        #self.model = YOTMLLP(self.batch_size, self.seq_len).to(self.device)
        self.model = YOTMCLP(self.batch_size, self.seq_len).to(self.device)
        #self.model = YOTMCLS(self.batch_size, self.seq_len).to(self.device)

        self.loss = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        self.model.load_checkpoint(self.model, self.optimizer, self.check_path)

        self.model.train()  # Set in training mode
        
    def post_proc(self):
        print("Model", self.model)
        self.model.save_weights(self.model, self.weights_path)

    def initialize_proc(self, epoch):
        self.sum_loss = 0

    def finalize_proc(self, epoch):
        self.TotalLoss.append(self.sum_loss)        
        print("Total Loss", self.TotalLoss)

        self.model.save_checkpoint(self.model, self.optimizer, self.check_path)

def main(argvs):
    train = Train(argvs)
    train.proc()

if __name__=='__main__':
    main('')
