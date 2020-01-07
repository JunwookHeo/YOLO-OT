import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from coord_utils import *

from YOTM import *

class LLPMPMParam:
    InfiSize = 128*52*52
    OutfiSize = 16*13*13
    OutCnnSize = OutfiSize
    LocSize = 5
    LocMapSize = 32
    InLstmSize = LocMapSize*LocMapSize
    HiddenSize = 1024 #4096
    LayerSize = 1
    OutputSize = LocMapSize*LocMapSize

class YimgNet(nn.Module):
    def __init__(self, batch_size, seq_len):
        super(YimgNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=1)

        self.lstm = nn.LSTM(input_size=LLPMPMParam.OutCnnSize, hidden_size=LLPMPMParam.HiddenSize, 
                            num_layers=LLPMPMParam.LayerSize, batch_first=True)
        self.hidden = self.init_hidden()

    def init_hidden(self):
            return (Variable(torch.zeros(LLPMPMParam.LayerSize, self.batch_size, LLPMPMParam.HiddenSize)), 
                Variable(torch.zeros(LLPMPMParam.LayerSize, self.batch_size, LLPMPMParam.HiddenSize)))

    def forward(self, x):
        batch_size, seq_size, C, H, W = x.size()
        x = x.view(batch_size*seq_size, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(batch_size, seq_size, -1)
        c_out, _ = self.lstm(x, self.hidden)
        
        return c_out


class LocNet(nn.Module):
    def __init__(self, batch_size, seq_len):
        super(LocNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        self.lstm = nn.LSTM(input_size=LLPMPMParam.InLstmSize, hidden_size=LLPMPMParam.HiddenSize, 
                            num_layers=LLPMPMParam.LayerSize, batch_first=True)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
            return (Variable(torch.zeros(LLPMPMParam.LayerSize, self.batch_size, LLPMPMParam.HiddenSize)), 
                Variable(torch.zeros(LLPMPMParam.LayerSize, self.batch_size, LLPMPMParam.HiddenSize)))

    def forward(self, x):
        batch_size, seq_size, N = x.size()
        x = x.view(batch_size * seq_size, -1)

        out = coord_utils.locations_to_probability_maps(LLPMPMParam.LocMapSize, x)
        out = out.view(batch_size, seq_size, -1)

        c_out, _ = self.lstm(out, self.hidden)
        
        return c_out


class YOTMLLP_PM(YOTM):
    def __init__(self, batch_size, seq_len):
        super(YOTMLLP_PM, self).__init__()
        self.yimgnet = YimgNet(batch_size, seq_len)
        self.locnet = LocNet(batch_size, seq_len)
        self.fc = nn.Linear(LLPMPMParam.HiddenSize, LLPMPMParam.OutputSize)
     
    def forward(self, x, l):
        c_x = self.yimgnet(x)
        c_l = self.locnet(l)
        c_out = c_x + c_l
        c_out = self.fc(c_out)
        return c_out
        
    def get_targets(self, targets):
        return coord_utils.locations_to_probability_maps(LLPMPMParam.LocMapSize, targets)

    def get_location(self, pm):
        return coord_utils.probability_map_to_location(LLPMPMParam.LocMapSize, pm)

    def save_checkpoint(self, model, optimizer, path):
        super().save_checkpoint(model, optimizer, path, 'yotmllp_pm.pth')

    def load_checkpoint(self, model, optimizer, path):
        super().load_checkpoint(model, optimizer, path, 'yotmllp_pm.pth')

    def save_weights(self, model, path):
        super().save_weights(model, path, 'yotmllp_pm.weights')

    def load_weights(self, model, path):
        super().load_weights(model, path, 'yotmllp_pm.weights')

