import torch
import torch.nn as nn
from torch.autograd import Variable

from YOTMWOPM import *

class RoloNet(nn.Module):
    def __init__(self, device, batch_size, seq_len, np):
        super(RoloNet, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.np = np

        self.lstm = nn.LSTM(input_size=self.np.InLstmSize, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, batch_first=True)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize).to(self.device)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize).to(self.device)))

    def forward(self, x, l):
        batch_size, seq_size, N = x.size()
        x = x.view(batch_size, seq_size, N)

        c_out, _ = self.lstm(x, self.hidden)

        return c_out[:,:,self.np.OutCnnSize + 1:-1]

class YOTMROLO(YOTMWOPM):
    class NP:
        InfiSize = 4096
        OutfiSize = 4*4
        OutCnnSize = 4096
        LocSize = 6
        LocMapSize = 32*32
        InLstmSize = OutCnnSize + LocSize
        HiddenSize = InLstmSize
        LayerSize = 1
        OutputSize = 4
        
    def __init__(self, batch_size, seq_len):
        super(YOTMROLO, self).__init__()
        self.np = self.NP()
        self.rolonet = RoloNet(self.device, batch_size, seq_len, self.np)

    def forward(self, x, l):
        out = self.rolonet(x, l)

        return out

