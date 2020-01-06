import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from YOTM import *

class ONELMParam:
    LocSize = 5
    LocMapSize = 32*32
    HiddenSize = 512 #4096
    LayerSize = 1
    OutputSize = 4

class OneLNet(nn.Module):
    def __init__(self, batch_size, seq_len):
        super(OneLNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size=ONELMParam.LocSize, hidden_size=ONELMParam.HiddenSize, 
                            num_layers=ONELMParam.LayerSize, batch_first=True)
        self.hidden = self.init_hidden()

        self.fc = nn.Linear(ONELMParam.HiddenSize, ONELMParam.OutputSize)
        
    def init_hidden(self):
            return (Variable(torch.zeros(ONELMParam.LayerSize, self.batch_size, ONELMParam.HiddenSize)), 
                Variable(torch.zeros(ONELMParam.LayerSize, self.batch_size, ONELMParam.HiddenSize)))

    def forward(self, x, l):
        batch_size, seq_size, N = l.size()
        l = l.view(batch_size, seq_size, -1)
        c_out, _ = self.lstm(l, self.hidden)
            
        c_out = self.fc(c_out)
        return c_out


class YOTMONEL(YOTM):
    def __init__(self, batch_size, seq_len):
        super(YOTMONEL, self).__init__()
        self.lstmlnet = OneLNet(batch_size, seq_len)
     
    def forward(self, x, l):
        batch_size, seq_size, _ = l.size()        
        out = self.lstmlnet(x, l)
        '''
        out = out.view(batch_size * seq_size, -1)
        l = l.view(l.size(0) * l.size(1), -1)
        t = torch.split(l, 4, dim=1)
        out = t[0]*t[1] + out*(1-t[1])
        out = out.view(batch_size, seq_size, -1)
        '''
        return out
        
    def save_checkpoint(self, model, optimizer, path):
        super().save_checkpoint(model, optimizer, path, 'yotmonel.pth')

    def load_checkpoint(self, model, optimizer, path):
        super().load_checkpoint(model, optimizer, path, 'yotmonel.pth')

    def save_weights(self, model, path):
        super().save_weights(model, path, 'yotmonel.weights')

    def load_weights(self, model, path):
        super().load_weights(model, path, 'yotmonel.weights')

