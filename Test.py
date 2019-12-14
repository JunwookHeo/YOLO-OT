import torch

from YOT_Base import YOT_Base
from YOTM import *

class Test(YOT_Base):
    def __init__(self,argvs = []):
        super(Test, self).__init__(argvs)
        # The path of dataset
        self.path = "data" 

    def post_proc(self, pos, frames, fis, locs, labels):
        with torch.no_grad():
            outputs = self.model(fis.float(), locs.float())
            for i, (frame, output) in enumerate(zip(frames, outputs)):
                outputs[i] = self.normal_to_locations(frame.shape[0], frame.shape[1], output.clamp(min=0))

            print(pos, outputs, labels)
            iou = self.bbox_iou(outputs, labels, False)
            print("IOU : ", iou)

    
    def pre_proc(self):
        self.model = YOTM(self.batch_size, self.seq_len).to(self.device)
        self.model.eval()  # Set in evaluation mode


def main(argvs):
    test = Test(argvs)
    test.run()

if __name__=='__main__':
    main('')
