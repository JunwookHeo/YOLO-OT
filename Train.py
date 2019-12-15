import torch
import torch.nn.functional as F

from YOT_Base import YOT_Base
from YOTM import *

class Train(YOT_Base):
    def __init__(self,argvs = []):
        super(Train, self).__init__(argvs)
        self.log_interval = 1
        # The path of dataset
        self.path = "data" 

    def post_proc(self, pos, frames, fis, locs, labels):
        for i, (frame, label) in enumerate(zip(frames, labels)):
            for j, (f, l) in enumerate(zip(frame, label)):
                labels[i][j] = self.locations_to_normal(f.shape[0], f.shape[1], l)
        
        outputs = self.model(fis.float(), locs.float())

        predicts = []
        targets = []
        for i, (output, label) in enumerate(zip(outputs, labels)):
            p = output[-1]
            l = label[-1]
            predicts.append(p)
            targets.append(l)

        loss = self.loss(torch.stack(predicts, dim=0), torch.stack(targets, dim=0))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        if pos % self.log_interval == 0:
            for p, t in zip(predicts, targets):
                print(p, t)
            print('Train pos: {} [Loss: {:.6f}]'.format(pos, loss.data))

    
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
