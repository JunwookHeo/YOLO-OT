import torch

from torch.autograd import Variable
from ListContainer import *

class Demo:
    def __init__(self,argvs = []):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = 6
        self.input_size = 1024
        self.hidden_size = 256
        self.output_size = 6
        self.num_sequence = 6
        self.num_layers =3
        self.img_size = 416

        self.path = "data"

    def run(self):
        
        for epoch in range(1):
            listContainer = ListContainer(self.path, self.batch_size, self.num_sequence, self.img_size)
            for dataLoader in listContainer:
                print(dataLoader)
                for frames, imgs, locs, labels in dataLoader:
                    imgs = Variable(imgs.to(self.device))
                    
                    self.post_proc(frames, imgs, locs, labels)
                    
    def post_proc(self, frames, imgs, locs, labels):
        loc = self.normal_to_location(frames[0].shape[0], frames[0].shape[1], locs[0][0:5])
        print(loc, labels[0])
        pass
    
    def normal_to_location(self, wid, ht, locations):
        #print("location in func: ", locations)
        wid *= 1.0
        ht *= 1.0
        locations[0] *= wid
        locations[1] *= ht
        locations[2] *= wid
        locations[3] *= ht
        return locations


def main(argvs):
    demo = Demo(argvs)
    demo.run()

if __name__=='__main__':
    main('')
