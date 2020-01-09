import os
import torch
import torch.nn as nn
from torch.autograd import Variable

class YOTM(nn.Module):
    def __init__(self):
        super(YOTM, self).__init__()
     
    def forward(self, x, l):
        pass
        
    def save_checkpoint(self, model, optimizer, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        file = os.path.join(path, name)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, file)
        
    def load_checkpoint(self, model, optimizer, path, name):
        file = os.path.join(path, name)
        if os.path.exists(file):
            checkpoint = torch.load(file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_weights(self, model, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        file = os.path.join(path, name)
        if os.path.exists(file):
            os.rename(file, file+'.bak')
        torch.save(model.state_dict(), file)


    def load_weights(self, model, path, name):
        file = os.path.join(path, name)
        if os.path.exists(file):
            model.load_state_dict(torch.load(file), strict=False)
            print(model)

    def get_targets(self, targets):
        return targets
    
    def get_location(self, pm):
        return pm

    def get_loss_function(self):
        return nn.MSELoss(reduction='sum')

