import os
from VideoDataset import *
from RoloDataset import *

class VideoLoader:
    @staticmethod
    def getDataset(path, label, seq_num, img_size):
        return VideoDataset(path, label, seq_num, img_size)

class RoloLoader:
    @staticmethod
    def getDataset(path, label, seq_num, img_size):
        return RoloDataset(path, label, seq_num, img_size)

class ListContainer:
    """ Loading folders which contain datasets """
    def __init__(self, path, batch_size, seq_num, img_size):
        self.pos = 0
        self.path = path
        self.batch_size = batch_size
        self.seq_num = seq_num
        self.img_size = img_size

        paths = [os.path.join(path,fn) for fn in next(os.walk(path))[1]]
        paths = sorted(paths)
        if len(paths) == 2:
            l = paths[0].split(os.sep)[-1]
            v = paths[1].split(os.sep)[-1]
            if l.lower() == 'labels' and v.lower() == 'videos' :
                self.load_videos(paths)
                return
        
        self.load_rolo(paths)
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= len(self.lists):
            raise StopIteration
        
        pos = self.pos
        self.pos += 1

        if len(self.labels) == 0:
            label = None
        else:
            label = self.labels[pos]
            
        dataset = self.loader.getDataset(self.lists[pos], label, self.seq_num, self.img_size)
        dataLoader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        
        return dataLoader        

    def load_videos(self, paths):
        self.labels = [os.path.join(paths[0],fn) for fn in next(os.walk(paths[0]))[2]]
        self.lists = [os.path.join(paths[1],fn) for fn in next(os.walk(paths[1]))[2]]
        self.loader = VideoLoader

    def load_rolo(self, paths):
        self.labels = []
        self.lists = []
        for path in paths:
            self.labels.append(os.path.join(path,"groundtruth_rect.txt"))
            self.lists.append(path)
        
        self.loader = RoloLoader


    
    
    