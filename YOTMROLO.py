import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from YOTM import *

class ROLOMParam:
    InfiSize = 128*52*52
    OutfiSize = 4*4
    OutCnnSize = 1*52*52
    LocSize = 5
    LocMapSize = 32*32
    InLstmSize = OutCnnSize + LocSize
    HiddenSize = 16
    LayerSize = 1
    OutputSize = 4

class RoloNet(nn.Module):
    def __init__(self, batch_size, seq_len):
        super(RoloNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size=ROLOMParam.InLstmSize, hidden_size=ROLOMParam.HiddenSize, 
                            num_layers=ROLOMParam.LayerSize, batch_first=True)

        self.fc = nn.Linear(ROLOMParam.HiddenSize, ROLOMParam.OutputSize)
        
        self.conv = nn.Conv2d(128, 1, kernel_size=1)

        self.hidden = self.init_hidden()

    def init_hidden(self):
            return (Variable(torch.zeros(ROLOMParam.LayerSize, self.batch_size, ROLOMParam.HiddenSize)), 
                Variable(torch.zeros(ROLOMParam.LayerSize, self.batch_size, ROLOMParam.HiddenSize)))

    def forward(self, x, l):
        batch_size, seq_size, C, W, H= x.size()
        #x = torch.mean(x, dim=2)
        x = x.view(batch_size* seq_size, C, W, H)
        x = F.relu(self.conv(x))
        x_out = x.view(batch_size, seq_size, -1)

        l_out = l.view(batch_size, seq_size, -1)
        c_out = torch.cat((x_out, l_out), dim=2)
        c_out = c_out.view(batch_size, seq_size, -1)
        c_out, _ = self.lstm(c_out, self.hidden)

        c_out = self.fc(c_out)
        c_out = c_out.view(batch_size, seq_size, -1)
        return c_out

class YOTMROLO(YOTM):
    def __init__(self, batch_size, seq_len):
        super(YOTMROLO, self).__init__()
        self.rolonet = RoloNet(batch_size, seq_len)

    def forward(self, x, l):
        out = self.rolonet(x, l)

        #out = torch.split(out, ROLOMParam.HiddenSize-5, dim=2)
        #out = torch.split(out[1], 4, dim=2)
        #return out[0]

        return out
        
    def get_targets(self, targets):
        return targets
    
    def get_location(self, pm):
        return pm
        
    def save_checkpoint(self, model, optimizer, path):
        super().save_checkpoint(model, optimizer, path, 'yotmrolo.pth')

    def load_checkpoint(self, model, optimizer, path):
        super().load_checkpoint(model, optimizer, path, 'yotmrolo.pth')

    def save_weights(self, model, path):
        super().save_weights(model, path, 'yotmrolo.weights')

    def load_weights(self, model, path):
        super().load_weights(model, path, 'yotmrolo.weights')