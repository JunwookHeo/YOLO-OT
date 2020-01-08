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
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 4, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(4, 1, kernel_size=1)

        self.fc = nn.Linear(self.np.OutfiSize, self.np.OutCnnSize)

    def forward(self, x):
        batch_size, seq_size, C, H, W = x.size()
        c_in = x.view(batch_size*seq_size, C, H, W)
        c_in = torch.relu(self.conv1(c_in))
        c_in = torch.relu(self.conv2(c_in))
        c_in = torch.relu(self.conv3(c_in))
        c_in = torch.relu(self.conv4(c_in))
        c_in = torch.relu(self.conv5(c_in))
        c_in = c_in.view(batch_size*seq_size, -1)
        c_in = self.fc(c_in)
        c_out = c_in.view(batch_size, seq_size, -1)
        return c_out

class LstmNet(nn.Module):
    def __init__(self, batch_size, seq_len, np):
        super(LstmNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.np = np

        self.lstm = nn.LSTM(input_size=self.np.InLstmSize, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, batch_first=True)

        self.hidden = self.init_hidden()

    def init_hidden(self):
            return (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)))

    def forward(self, x):
        x = x.view(x.size(0), self.seq_len, -1)
        c_out, _ = self.lstm(x, self.hidden)
        return c_out

class YOTMCLP(YOTM):
    class NP:
        InfiSize = 128*52*52
        OutfiSize = 4*4 #64*26*26
        OutCnnSize = 4 #8192
        LocSize = 5
        LocMapSize = 32*32
        InLstmSize = LocSize
        HiddenSize = 4 #4096
        LayerSize = 1
        OutputSize = 4
        
    def __init__(self, batch_size, seq_len):
        super(YOTMCLP, self).__init__()
        self.np = self.NP()

        self.yimgnet = YimgNet(self.np)
        self.lstmnet = LstmNet(batch_size, seq_len, self.np)
        self.fc = nn.Linear(self.np.HiddenSize, self.np.OutputSize)
     
    def forward(self, x, l):
        x_out = self.yimgnet(x)
        l_out = self.lstmnet(l)
        c_out = x_out + l_out
        c_out = self.fc(c_out)
        c_out = c_out.view(c_out.size(0), c_out.size(1), -1)

        return c_out

    def save_checkpoint(self, model, optimizer, path):
        super().save_checkpoint(model, optimizer, path, 'yotmclp.pth')

    def load_checkpoint(self, model, optimizer, path):
        super().load_checkpoint(model, optimizer, path, 'yotmclp.pth')
 
    def save_weights(self, model, path):
        super().save_weights(model, path, 'yotmclp.weights')

    def load_weights(self, model, path):
        super().load_weights(model, path, 'yotmclp.weights')
