import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from YOTM import *

class YimgNet(nn.Module):
    def __init__(self, np):
        super(YimgNet, self).__init__()
        self.np = np

        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=1)

        self.link1 = nn.Linear(self.np.OutCnnSize, self.np.LocMapSize*2)
        self.link2 = nn.Linear(self.np.LocSize, self.np.LocMapSize*2)

    def forward(self, x, l):
        batch_size, seq_size, C, H, W = x.size()
        x = x.view(batch_size*seq_size, C, H, W)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = x.view(batch_size*seq_size, -1)
        x = torch.relu(self.link1(x))

        l = l.view(batch_size*seq_size, -1)
        l = torch.relu(self.link2(l))

        x = x.view(batch_size, seq_size, -1)
        l = l.view(batch_size, seq_size, -1)

        c_out = torch.cat((x, l), 2)

        return c_out

class LstmNet(nn.Module):
    def __init__(self, batch_size, seq_len, np):
        super(LstmNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.np = np

        self.lstm = nn.LSTM(input_size=self.np.InLstmSize, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, batch_first=True)

        # final fully connected layers
        self.dense = nn.Linear(self.np.HiddenSize, 128)
        self.fc = nn.Linear(128, self.np.OutputSize)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)))

    def forward(self, x):
        x, _ = self.lstm(x, self.hidden)
        c_out = torch.relu(self.dense(x)) 
        c_out = self.fc(c_out)

        return c_out

class YOTMCLP(YOTM):
    class NP:
        InfiSize = 128*52*52
        OutfiSize = 16*13*13
        OutCnnSize = OutfiSize
        LocSize = 5
        LocMapSize = 32*32
        InLstmSize = 4*LocMapSize
        HiddenSize = 2048
        LayerSize = 1
        OutputSize = 4
        
    def __init__(self, batch_size, seq_len):
        super(YOTMCLP, self).__init__()
        self.np = self.NP()

        self.yimgnet = YimgNet(self.np)
        self.lstmnet = LstmNet(batch_size, seq_len, self.np)
        
     
    def forward(self, x, l):
        c_out = self.yimgnet(x, l)
        c_out = self.lstmnet(c_out)

        return c_out

