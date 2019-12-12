# YOT_Base class
from torch.autograd import Variable
from ListContainer import *

class YOT_Base:
    def __init__(self,argvs = []):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = 1
        self.num_sequence = 6
        self.num_layers =3
        self.img_size = 416


    def run(self):
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        for epoch in range(1):
            listContainer = ListContainer(self.path, self.batch_size, self.num_sequence, self.img_size)
            for dataLoader in listContainer:
                pos = 0
                for frames, imgs, labels in dataLoader:
                    imgs = Variable(imgs.type(Tensor))
                    
                    self.post_proc(pos, frames, imgs, labels)
                    pos += 1
                    
    def post_proc(self, pos, frames, imgs, labels):
        loc = self.locations_normal(frames[0].shape[0], frames[0].shape[1], imgs[0][-5:-1])
        print(pos, loc, labels[0])
        pass
    
    def locations_normal(self, wid, ht, locations):
        #print("location in func: ", locations)
        wid *= 1.0
        ht *= 1.0
        locations[0] *= wid
        locations[1] *= ht
        locations[2] *= wid
        locations[3] *= ht
        return locations

    def load_dataset(self):
        pass
    
    def build_model(self):
        pass

