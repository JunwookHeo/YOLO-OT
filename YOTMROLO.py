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

        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=1)

        self.lstm = nn.LSTM(input_size=self.np.InLstmSize, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, batch_first=True)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize).to(self.device)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize).to(self.device)))

    def forward(self, x, l):
        batch_size, seq_size, C, W, H= x.size()
        x = x.view(batch_size*seq_size, C, H, W)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = x.view(batch_size, seq_size, -1)
        l = l.view(batch_size, seq_size, -1)
        c_out = torch.cat((x, l), dim=2)
        c_out, _ = self.lstm(c_out, self.hidden)

        return c_out[:,:,self.np.OutCnnSize:-1]

class YOTMROLO(YOTMWOPM):
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
        self.rolonet = RoloNet(self.device, batch_size, seq_len, self.np)

    def forward(self, x, l):
        out = self.rolonet(x, l)

        #out = torch.split(out, ROLOMParam.HiddenSize-5, dim=2)
        #out = torch.split(out[1], 4, dim=2)
        #return out[0]

        return out

