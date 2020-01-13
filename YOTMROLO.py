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

        self.fc = nn.Linear(self.np.HiddenSize, self.np.OutputSize)
        
        self.conv = nn.Conv2d(128, 1, kernel_size=1)

        self.hidden = self.init_hidden()

    def init_hidden(self):
            return (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)))

    def forward(self, x, l):
        batch_size, seq_size, C, W, H= x.size()
        #x = torch.mean(x, dim=2)
        x = x.view(batch_size* seq_size, C, W, H)
        x = torch.relu(self.conv(x))
        x_out = x.view(batch_size, seq_size, -1)

        l_out = l.view(batch_size, seq_size, -1)
        c_out = torch.cat((x_out, l_out), dim=2)
        c_out = c_out.view(batch_size, seq_size, -1)
        c_out, _ = self.lstm(c_out, self.hidden)

        c_out = self.fc(c_out)
        c_out = c_out.view(batch_size, seq_size, -1)
        return c_out

class YOTMROLO(YOTM):
    class NP:
        InfiSize = 128*52*52
        OutfiSize = 4*4
        OutCnnSize = 1*52*52
        LocSize = 5
        LocMapSize = 32*32
        InLstmSize = OutCnnSize + LocSize
        HiddenSize = 16
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

