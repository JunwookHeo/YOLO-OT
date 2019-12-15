import torch

from YOT_Base import YOT_Base
from YOTM import *

class Test(YOT_Base):
    def __init__(self,argvs = []):
        super(Test, self).__init__(argvs)
        # The path of dataset
        self.path = "../DATA" 

    def post_proc(self, epoch, pos, frames, fis, locs, labels):
        with torch.no_grad():
            outputs = self.model(fis.float(), locs.float())
            predicts = []
            targets = []
            for i, (frame, output, label) in enumerate(zip(frames, outputs, labels)):
                f = frame[-1]
                o = output[-1]
                l = label[-1]
                p = self.normal_to_locations(f.size(0), f.size(1), o.clamp(min=0))
                predicts.append(p)
                targets.append(l)

            for p, t in zip(predicts, targets):
                print(epoch, pos, p, t)
                
            iou = self.bbox_iou(torch.stack(predicts, dim=0),  torch.stack(targets, dim=0), False)            
            print("\tIOU : ", iou)

        return 0

    
    def pre_proc(self):
        self.model = YOTM(self.batch_size, self.seq_len).to(self.device)
        self.model.eval()  # Set in evaluation mode
        print(self.model)


def main(argvs):
    test = Test(argvs)
    test.run()

if __name__=='__main__':
    main('')
