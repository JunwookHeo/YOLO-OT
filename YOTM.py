import os
import torch
import torch.nn as nn
from torch.autograd import Variable

class YOTM(nn.Module):
    def __init__(self):
        super(YOTM, self).__init__()

    def forward(self, x, l):
        pass
        
    def __save_checkpoint(self, model, optimizer, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        file = os.path.join(path, name)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, file)
        
    def __load_checkpoint(self, model, optimizer, path, name):
        file = os.path.join(path, name)
        if os.path.exists(file):
            checkpoint = torch.load(file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def __save_weights(self, model, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        file = os.path.join(path, name)
        if os.path.exists(file):
            os.rename(file, file+'.bak')
        torch.save(model.state_dict(), file)


    def __load_weights(self, model, path, name):
        file = os.path.join(path, name)
        if os.path.exists(file):
            model.load_state_dict(torch.load(file), strict=False)

    def get_targets(self, targets):
        return targets
    
    def get_location(self, pm):
        return pm

    def get_loss_function(self):
        return nn.MSELoss(reduction='sum')

    def save_checkpoint(self, model, optimizer, path):
        self.__save_checkpoint(model, optimizer, path, self.__class__.__name__.lower() + '.pth')

    def load_checkpoint(self, model, optimizer, path):
        self.__load_checkpoint(model, optimizer, path, self.__class__.__name__.lower() + '.pth')
 
    def save_weights(self, model, path):
        self.__save_weights(model, path, self.__class__.__name__.lower() + '.weights')

    def load_weights(self, model, path):
        self.__load_weights(model, path, self.__class__.__name__.lower() + '.weights')
