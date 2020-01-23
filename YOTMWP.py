import torch
import torch.nn as nn
from torch.autograd import Variable
from YOTM import YOTM


class YimgNet(nn.Module):
    def __init__(self, batch_size, seq_len, np):
        super(YimgNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.np = np

        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=1)
        
    def forward(self, x, l):
        batch_size, seq_size, C, H, W = x.size()
        x = x.view(batch_size*seq_size, C, H, W)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
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
        
        self.lstm1 = nn.LSTM(input_size=self.np.InLstmSize, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.np.HiddenSize, hidden_size=128, 
                            num_layers=self.np.LayerSize, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=4, 
                            num_layers=self.np.LayerSize, batch_first=True)
        self.hidden1, self.hidden2, self.hidden1 = self.init_hidden()
        
    def init_hidden(self):
        hidden1 = (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)))
        hidden2 = (Variable(torch.zeros(self.np.LayerSize, self.batch_size, 128)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, 128)))
        hidden3 = (Variable(torch.zeros(self.np.LayerSize, self.batch_size, 4)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, 4)))
        return hidden1, hidden2, hidden3

    def forward(self, x):
        x, _ = self.lstm1(x, self.hidden1)
        x, _ = self.lstm2(x, self.hidden2)
        c_out, _ = self.lstm3(x, self.hidden3)
        
        return c_out


class YOTMWP(YOTM):
    class NP:
        InfiSize = 128*52*52
        OutfiSize = 16*13*13
        OutCnnSize = OutfiSize
        LocSize = 5
        LocMapSize = 32*32
        InLstmSize = OutCnnSize + LocSize
        HiddenSize = 2048 #4096
        LayerSize = 1
        OutputSize = 4
        
    def __init__(self, batch_size, seq_len):
        super(YOTMWP, self).__init__()
        self.np = self.NP()

        self.yimgnet = YimgNet(batch_size, seq_len, self.np)
        self.lstmnet = LstmNet(batch_size, seq_len, self.np)
        
    def forward(self, x, l):
        c_out = self.yimgnet(x, l)
        c_out = self.lstmnet(c_out)
        
        return c_out
    
