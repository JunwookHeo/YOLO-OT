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
        for i, (frame, l) in enumerate(zip(frames, labels)):
            labels[i] = self.locations_to_normal(frame.shape[0], frame.shape[1], l)

        
        outputs = self.model(fis.float(), locs.float())

        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    
        print(pos, outputs)
        if pos % self.log_interval == 0:
            print('Train pos: {} [Loss: {:.6f}]'.format(pos, loss.data))

    
    def pre_proc(self):        
        self.model = YOTM(self.batch_size, self.seq_len).to(self.device)

        self.loss = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        self.model.train()  # Set in training mode


def main(argvs):
    train = Train(argvs)
    train.run()

if __name__=='__main__':
    main('')
