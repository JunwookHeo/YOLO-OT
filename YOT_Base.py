# YOT_Base class
from torch.autograd import Variable
from ListContainer import *

class YOT_Base:
    def __init__(self,argvs = []):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = 6
        self.seq_len = 1
        self.img_size = 416


    def run(self):
        self.pre_proc()

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        for epoch in range(1):
            listContainer = ListContainer(self.path, self.batch_size, self.seq_len, self.img_size)
            for dataLoader in listContainer:
                pos = 0
                for frames, fis, locs, labels in dataLoader:
                    fis = Variable(fis.type(Tensor))
                    locs = Variable(locs.type(Tensor))
                    labels = Variable(labels.type(Tensor))
                    
                    self.post_proc(pos, frames, fis, locs, labels)
                    pos += 1
    
    def pre_proc(self):
        pass

    def post_proc(self, pos, frames, fis, locs, labels):
        for frame, loc, label in zip(frames, locs, labels):
            loc = self.normal_to_locations(frame.shape[0], frame.shape[1], loc)
            print(pos, loc, label)
        
    
    def normal_to_locations(self, wid, ht, locations):
        #print("location in func: ", locations)
        wid *= 1.0
        ht *= 1.0
        locations[0] *= wid
        locations[1] *= ht
        locations[2] *= wid
        locations[3] *= ht
        return locations

    def locations_to_normal(self, wid, ht, locations):
        #print("location in func: ", locations)
        wid *= 1.0
        ht *= 1.0
        locations[0] /= wid
        locations[1] /= ht
        locations[2] /= wid
        locations[3] /= ht
        return locations

    def load_dataset(self):
        pass
    
    def build_model(self):
        pass

