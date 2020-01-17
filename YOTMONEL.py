import torch
import torch.nn as nn
from torch.autograd import Variable

from YOTM import *

class OneLNet(nn.Module):
    def __init__(self, batch_size, seq_len, np):
        super(OneLNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.np = np

        self.lstm = nn.LSTM(input_size=self.np.LocSize, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, dropout=0.3, batch_first=True)
        self.init_hidden()

        self.fc = nn.Linear(self.np.HiddenSize, self.np.OutputSize)
        
    def init_hidden(self):
        self.hidden = (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)))

    def forward(self, x, l):
        batch_size, seq_size, N = l.size()
        l = l.view(batch_size, seq_size, -1)
        c_out, _ = self.lstm(l, self.hidden)
            
        c_out = self.fc(c_out)
        return c_out


class YOTMONEL(YOTM):
    class NP:
        LocSize = 5
        LocMapSize = 32*32
        HiddenSize = 32 #4096
        LayerSize = 1
        OutputSize = 4
        
    def __init__(self, batch_size, seq_len):
        super(YOTMONEL, self).__init__()
        self.np = self.NP()

        self.lstmlnet = OneLNet(batch_size, seq_len, self.np)
     
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
    
    def init_hidden(self):
        self.lstmlnet.init_hidden()
    