import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from YOTM import *

class CLSMPMParam:
    InfiSize = 128*52*52
    OutfiSize = 16*16
    OutCnnSize = 16*16 #8192
    LocSize = OutCnnSize
    LocMapSize = 32*32
    InLstmSize = OutCnnSize #+ LocSize
    HiddenSize = OutCnnSize #4096
    LayerSize = 1
    OutputSize = OutCnnSize


class YimgNet_PM(nn.Module):
    def __init__(self):
        super(YimgNet_PM, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(CLSMPMParam.OutfiSize, CLSMPMParam.OutCnnSize)

    def forward(self, x, l):
        batch_size, seq_size, C, H, W = x.size()
        c_in = x.view(batch_size*seq_size, C, H, W)
        c_in = F.relu(self.conv1(c_in))
        c_in = F.relu(self.conv2(c_in))
        c_in = F.relu(self.conv3(c_in))
        c_in = F.relu(self.conv4(c_in))
        c_in = c_in.view(batch_size*seq_size, -1)
        c_in = F.relu(self.fc(c_in))
        c_out = c_in.view(batch_size, seq_size, -1)
        #c_out = torch.cat((c_out,l), 2)
        l_out = l.view(batch_size, seq_size, -1)
        c_out = c_out + l_out
        return c_out

class LstmNet_PM(nn.Module):
    def __init__(self, batch_size, seq_len):
        super(LstmNet_PM, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size=CLSMPMParam.InLstmSize, hidden_size=CLSMPMParam.HiddenSize, 
                            num_layers=CLSMPMParam.LayerSize, batch_first=True)

        self.fc = nn.Linear(CLSMPMParam.HiddenSize, CLSMPMParam.OutputSize)
        self.hidden = self.init_hidden()

    def init_hidden(self):
            return (Variable(torch.zeros(CLSMPMParam.LayerSize, self.batch_size, CLSMPMParam.HiddenSize)), 
                Variable(torch.zeros(CLSMPMParam.LayerSize, self.batch_size, CLSMPMParam.HiddenSize)))

    def forward(self, x):
        x = x.view(x.size(0), self.seq_len, -1)
        c_out, _ = self.lstm(x, self.hidden)
        c_out = self.fc(c_out)
        return c_out

class YOTMCLS_PM(YOTM):
    def __init__(self, batch_size, seq_len):
        super(YOTMCLS_PM, self).__init__()
        self.yimgnet = YimgNet_PM()
        self.lstmnet = LstmNet_PM(batch_size, seq_len)
        
    def forward(self, x, l):
        out = self.yimgnet(x, l)
        out = self.lstmnet(out)
        out = out.view(out.size(0), out.size(1), -1)

        return out
        
    def save_checkpoint(self, model, optimizer, path):
        super().save_checkpoint(model, optimizer, path, 'yotmcls.pth')

    def load_checkpoint(self, model, optimizer, path):
        super().load_checkpoint(model, optimizer, path, 'yotmcls.pth')

    def save_weights(self, model, path):
        super().save_weights(model, path, 'yotmcls.weights')

    def load_weights(self, model, path):
        super().load_weights(model, path, 'yotmcls.weights')