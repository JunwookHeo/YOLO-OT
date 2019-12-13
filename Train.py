import torch
import torch.nn.functional as F

from YOT_Base import YOT_Base
from YOTM import *

class Train(YOT_Base):
    def __init__(self,argvs = []):
        super(Train, self).__init__(argvs)
        self.log_interval = 10
        # The path of dataset
        self.path = "data" 

    def post_proc(self, pos, frames, fis, locs, labels):
        self.optimizer.zero_grad()
        
        output = self.model(fis.float(), locs.float())

        loss = self.loss(output, labels)
        loss.backward()
        self.optimizer.step()
    
        print(pos, output)
        if pos % self.log_interval == 0:
            print('Train pos: {} [Loss: {:.6f}'.format(
                pos, loss.data))

    
    def pre_proc(self):        
        self.model = YOTM(self.batch_size, self.seq_len).to(self.device)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        self.model.train()  # Set in training mode


def main(argvs):
    train = Train(argvs)
    train.run()

if __name__=='__main__':
    main('')
