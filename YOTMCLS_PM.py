import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from YOTM import *

class CLSMPMParam:
    InfiSize = 128*52*52
    OutfiSize = 16*13*13
    OutCnnSize = OutfiSize
    LocSize = 5
    LocMapSize = 32
    InLstmSize = OutCnnSize + LocMapSize*LocMapSize
    HiddenSize = 2048 
    LayerSize = 1
    OutputSize = LocMapSize*LocMapSize


class YimgNet_PM(nn.Module):
    def __init__(self):
        super(YimgNet_PM, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=1)

    def forward(self, x, l):
        batch_size, seq_size, C, H, W = x.size()
        x = x.view(batch_size*seq_size, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(batch_size, seq_size, -1)
        l = l.view(batch_size, seq_size, -1)
        c_out = torch.cat((x, l), 2)
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

        return out
        
    def save_checkpoint(self, model, optimizer, path):
        super().save_checkpoint(model, optimizer, path, 'yotmcls_pm.pth')

    def load_checkpoint(self, model, optimizer, path):
        super().load_checkpoint(model, optimizer, path, 'yotmcls_pm.pth')

    def save_weights(self, model, path):
        super().save_weights(model, path, 'yotmcls_pm.weights')

    def load_weights(self, model, path):
        super().load_weights(model, path, 'yotmcls_pm.weights')