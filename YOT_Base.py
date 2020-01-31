# YOT_Base class

from abc import ABC, abstractmethod
from torch.autograd import Variable
from ListContainer import ListContainer

import torch
import argparse

class YOT_Base(ABC):
    def __init__(self,argvs = []):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        opt = self.parse_default_config()

        self.data_path = opt.data_path
        self.dataset = opt.dataset
        self.batch_size = opt.batch_size
        self.seq_len = opt.sequence_length
        self.img_size = opt.img_size
        self.epochs = opt.epochs
        
        self.weights_path = opt.weights_path
        self.save_weights = opt.save_weights

        self.mode = opt.run_mode
        self.model_name = opt.model_name

    def parse_default_config(self):
        parser = argparse.ArgumentParser()

        # default argument
        parser.add_argument("--data_path", type=str, default="../rolo_data", help="path to data config file")
        parser.add_argument("--dataset", type=str, default="yot", help="sort of dataset. yot, rolo or video")
    
        parser.add_argument("--epochs", type=int, default=30, help="size of epoch")
        parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
        parser.add_argument("--sequence_length", type=int, default=6, help="size of each sequence of LSTM")
        parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
        
        parser.add_argument("--weights_path", type=str, default="outputs/weights", help="path to weights folder")
        parser.add_argument("--save_weights", type=bool, default=False, help="save checkpoint and weights")

        parser.add_argument("--run_mode", type=str, default="none", help="train, validate or test mode")        
        parser.add_argument("--model_name", type=str, default="YOTMLLP", help="class name of the model")

        args, _ = parser.parse_known_args()
        return args

    def proc(self):
        self.pre_proc()

        for epoch in range(self.epochs):            
            self.initialize_epoch_processing(epoch)
            listContainer = ListContainer(self.dataset, self.data_path, self.batch_size, self.seq_len, self.img_size, self.mode)
            for lpos, dataLoader in enumerate(listContainer):
                path = listContainer.get_list_info(lpos)
                self.initialize_list_loop(path)
                for dpos, (frames, fis, locs, labels) in enumerate(dataLoader):
                    fis = Variable(fis.to(self.device))
                    locs = Variable(locs.to(self.device))
                    labels = Variable(labels.to(self.device), requires_grad=False)

                    self.processing(epoch, lpos, dpos, frames, fis, locs, labels)
                self.finalize_list_loop()
            self.finalize_epoch_processing(epoch)
        
        self.post_proc()

    @abstractmethod
    def pre_proc(self):
        raise NotImplementedError

    @abstractmethod
    def processing(self, epoch, lpos, dpos, frames, fis, locs, labels):
        raise NotImplementedError

    @abstractmethod
    def post_proc(self):
        raise NotImplementedError
    
    @abstractmethod
    def initialize_list_loop(self, name):
        raise NotImplementedError

    @abstractmethod
    def finalize_list_loop(self):
        raise NotImplementedError

    @abstractmethod
    def initialize_epoch_processing(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def finalize_epoch_processing(self, epoch):
        raise NotImplementedError
    
    def get_last_sequence(self, data):
        d = torch.split(data, self.seq_len -1, dim=1)
        return torch.squeeze(d[1], dim=1)

    
