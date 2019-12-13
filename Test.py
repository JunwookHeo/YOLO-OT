import torch

from YOT_Base import YOT_Base
from YOTM import *

class Test(YOT_Base):
    def __init__(self,argvs = []):
        super(Test, self).__init__(argvs)
        # The path of dataset
        self.path = "data" 

    def post_proc(self, pos, frames, fis, locs, labels):
        output = self.model(fis.float(), locs.float())
        print(pos, output)

    
    def pre_proc(self):
        self.model = YOTM(self.batch_size, self.seq_len).to(self.device)
        self.model.eval()  # Set in evaluation mode


def main(argvs):
    test = Test(argvs)
    test.run()

if __name__=='__main__':
    main('')
