import torch
import torch.nn as nn
from torch.autograd import Variable

from YOTM import *

class YimgNet(nn.Module):
    def __init__(self, batch_size, seq_len, np):
        super(YimgNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.np = np

        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=1)
        
        self.lstm = nn.LSTM(input_size=self.np.OutCnnSize, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, batch_first=True)
        self.init_hidden()

    def init_hidden(self):
        self.hidden = (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)))

    def forward(self, x):
        batch_size, seq_size, C, H, W = x.size()
        x = x.view(batch_size*seq_size, C, H, W)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = x.view(batch_size, seq_size, -1)
        c_out, _ = self.lstm(x, self.hidden)
        
        return c_out


class LocNet(nn.Module):
    def __init__(self, batch_size, seq_len, np):
        super(LocNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.np = np
        
        self.lstm = nn.LSTM(input_size=self.np.LocSize, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, batch_first=True)
        self.init_hidden()
        
    def init_hidden(self):
        self.hidden = (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)))

    def forward(self, l):
        #batch_size, seq_size, N, W, H = x.size()
        #x = x.view(batch_size, seq_size, -1)
        
        c_out, _ = self.lstm(l, self.hidden)
        
        return c_out


class YOTMLLP(YOTM):
    class NP:
        InfiSize = 128*52*52
        OutfiSize = 16*13*13
        OutCnnSize = OutfiSize
        LocSize = 5
        InLstmSize = OutCnnSize
        HiddenSize = 1024 #4096
        LayerSize = 1
        OutputSize = 4
        
    def __init__(self, batch_size, seq_len):
        super(YOTMLLP, self).__init__()
        self.np = self.NP()

        self.yimgnet = YimgNet(batch_size, seq_len, self.np)
        self.locnet = LocNet(batch_size, seq_len, self.np)
        self.fc = nn.Linear(self.np.HiddenSize, self.np.OutputSize)
     
    def forward(self, x, l):
        c_x = self.yimgnet(x)
        c_l = self.locnet(l)
        c_out = c_x + c_l
        c_out = self.fc(c_out)
        return c_out
    
    def init_hidden(self):
        self.locnet.init_hidden()
        self.yimgnet.init_hidden()
    