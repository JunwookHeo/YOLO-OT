import torch

from YOT_Base import YOT_Base
from YOTMCLP import *

class Test(YOT_Base):
    def __init__(self,argvs = []):
        super(Test, self).__init__(argvs)
        # The path of dataset
        self.path = "../rolo_data" 

        self.epochs = 1

    def processing(self, epoch, pos, frames, fis, locs, labels):
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
        self.model = YOTMCLP(self.batch_size, self.seq_len).to(self.device)
        self.model.load_weights(self.model, self.weights_path)
        self.model.eval()  # Set in evaluation mode

        print(self.model)

    def post_proc(self):
        pass

    def initialize_proc(self, epoch):
        pass

    def finalize_proc(self, epoch):
        pass

def main(argvs):
    test = Test(argvs)
    test.proc()

if __name__=='__main__':
    main('')
