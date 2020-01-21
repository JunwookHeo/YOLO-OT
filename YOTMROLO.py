import torch
import torch.nn as nn
from torch.autograd import Variable

from YOTM import *

class RoloNet(nn.Module):
    def __init__(self, batch_size, seq_len, np):
        super(RoloNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.np = np

        self.lstm = nn.LSTM(input_size=self.np.InLstmSize, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, batch_first=True)

        self.init_hidden()

    def init_hidden(self):
        self.hidden = (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)))

    def forward(self, x, l):
        batch_size, seq_size, C, W, H= x.size()
        
        x = x.view(batch_size, seq_size, -1)
        l = l.view(batch_size, seq_size, -1)
        c_out = torch.cat((x, l), dim=2)
        c_out, self.hidden = self.lstm(c_out, self.hidden)

        return c_out[:,:,self.np.OutCnnSize:]

class YOTMROLO(YOTM):
    class NP:
        InfiSize = 128*52*52
        OutfiSize = 4*4
        OutCnnSize = 1*52*52
        LocSize = 5
        LocMapSize = 32*32
        InLstmSize = OutCnnSize + LocSize
        HiddenSize = InLstmSize
        LayerSize = 1
        OutputSize = 4
        
    def __init__(self, batch_size, seq_len):
        super(YOTMROLO, self).__init__()
        self.np = self.NP()
        self.rolonet = RoloNet(batch_size, seq_len, self.np)

    def forward(self, x, l):
        out = self.rolonet(x, l)

        #out = torch.split(out, ROLOMParam.HiddenSize-5, dim=2)
        #out = torch.split(out[1], 4, dim=2)
        #return out[0]

        return out

    def init_hidden(self):
        self.rolonet.init_hidden()
    
