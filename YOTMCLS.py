import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from YOTM import *

class CLSMParam:
    InfiSize = 128*52*52
    OutfiSize = 16*13*13
    OutCnnSize = OutfiSize
    LocSize = 5
    LocMapSize = 32*32
    InLstmSize = OutCnnSize + LocSize
    HiddenSize = 2048
    LayerSize = 1
    OutputSize = 4


class YimgNet(nn.Module):
    def __init__(self):
        super(YimgNet, self).__init__()
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
        c_out = torch.cat((x, l), 2)
        return c_out

class LstmNet(nn.Module):
    def __init__(self, batch_size, seq_len):
        super(LstmNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size=CLSMParam.InLstmSize, hidden_size=CLSMParam.HiddenSize, 
                            num_layers=CLSMParam.LayerSize, batch_first=True)

        self.fc = nn.Linear(CLSMParam.HiddenSize, CLSMParam.OutputSize)
        self.hidden = self.init_hidden()

    def init_hidden(self):
            return (Variable(torch.zeros(CLSMParam.LayerSize, self.batch_size, CLSMParam.HiddenSize)), 
                Variable(torch.zeros(CLSMParam.LayerSize, self.batch_size, CLSMParam.HiddenSize)))

    def forward(self, x):
        c_out, _ = self.lstm(x, self.hidden)
        c_out = self.fc(c_out)
        return c_out

class YOTMCLS(YOTM):
    def __init__(self, batch_size, seq_len):
        super(YOTMCLS, self).__init__()
        self.yimgnet = YimgNet()
        self.lstmnet = LstmNet(batch_size, seq_len)
        
    def forward(self, x, l):
        out = self.yimgnet(x, l)
        out = self.lstmnet(out)
        #out = out.view(out.size(0), out.size(1), -1)

        return out
        
    def save_checkpoint(self, model, optimizer, path):
        super().save_checkpoint(model, optimizer, path, 'yotmcls.pth')

    def load_checkpoint(self, model, optimizer, path):
        super().load_checkpoint(model, optimizer, path, 'yotmcls.pth')

    def save_weights(self, model, path):
        super().save_weights(model, path, 'yotmcls.weights')

    def load_weights(self, model, path):
        super().load_weights(model, path, 'yotmcls.weights')