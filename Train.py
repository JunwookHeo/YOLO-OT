import torch
import torch.nn.functional as F

from YOT_Base import YOT_Base
from YOTM import *

class Train(YOT_Base):
    def __init__(self,argvs = []):
        super(Train, self).__init__(argvs)
        self.log_interval = 1
        # The path of dataset
        self.path = "../DATA" 

    def post_proc(self, epoch, pos, frames, fis, locs, labels):
        outputs = self.model(fis.float(), locs.float())

        predicts = []
        targets = []
        for i, (f, output, label) in enumerate(zip(frames, outputs, labels)):
            p = output[-1]
            l = label[-1]
            p = self.normal_to_locations(f.shape[1], f.shape[2], p)
            predicts.append(p)
            targets.append(l)

        loss = self.loss(torch.stack(predicts, dim=0), torch.stack(targets, dim=0))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        if pos % self.log_interval == 0:
            print('Train pos: {}-{} [Loss: {:.6f}]'.format(epoch, pos, loss.data))

            for p, t in zip(predicts, targets):
                print("\t", p, t)
            iou = self.bbox_iou(torch.stack(predicts, dim=0),  torch.stack(targets, dim=0), False)            
            print("\tIOU : ", iou)


    
    def pre_proc(self):        
        self.model = YOTM(self.batch_size, self.seq_len).to(self.device)

        self.loss = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        self.model.train()  # Set in training mode


def main(argvs):
    train = Train(argvs)
    train.run()

if __name__=='__main__':
    main('')
