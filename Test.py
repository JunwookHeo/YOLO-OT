import torch

from YOT_Base import YOT_Base
from YOTM import *

class Test(YOT_Base):
    def __init__(self,argvs = []):
        super(Test, self).__init__(argvs)

        self.path = "data"

    def post_proc(self, pos, frames, imgs, labels):
        for img in imgs:
            fi = img[0:128*52*52].reshape(1, 128, 52, 52).double()
            fo = fi.float()
            output = self.model(fi.float())
            print(pos, output)
    
    def test(self):
        # TODO: CNN to CNN + LSTM with a variable step number
        self.model = CNN().to(self.device)
        self.model.eval()  # Set in evaluation mode


def main(argvs):
    test = Test(argvs)
    test.test()
    test.run()

if __name__=='__main__':
    main('')
