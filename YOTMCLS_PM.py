import torch
import torch.nn as nn
from torch.autograd import Variable

from coord_utils import *

from YOTMWPM import *

class YimgNet_PM(nn.Module):
    def __init__(self, np):
        super(YimgNet_PM, self).__init__()
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
        
        l = l.view(batch_size * seq_size, -1)
        l = coord_utils.locations_to_probability_maps(self.np.LocMapSize, l)
        l = l.view(batch_size, seq_size, -1)

        c_out = torch.cat((x, l), 2)
        return c_out

class LstmNet_PM(nn.Module):
    def __init__(self, device, batch_size, seq_len, np):
        super(LstmNet_PM, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.np = np

        self.lstm = nn.LSTM(input_size=self.np.InLstmSize, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, batch_first=True)

        self.fc = nn.Linear(self.np.HiddenSize, self.np.OutputSize)
        
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize).to(self.device)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize).to(self.device)))

    def forward(self, x):
        c_out, _ = self.lstm(x, self.hidden)
        c_out = self.fc(c_out)
        return c_out

class YOTMCLS_PM(YOTMWPM):
    class NP:
        InfiSize = 128*52*52
        OutfiSize = 16*13*13
        OutCnnSize = OutfiSize
        LocSize = 5
        LocMapSize = 32
        InLstmSize = OutCnnSize + LocMapSize*LocMapSize
        HiddenSize = 2048 
        LayerSize = 1
        OutputSize = LocMapSize*LocMapSize
        
    def __init__(self, batch_size, seq_len):
        super(YOTMCLS_PM, self).__init__()
        self.np = self.NP()

        self.yimgnet = YimgNet_PM(self.np)
        self.lstmnet = LstmNet_PM(self.device, batch_size, seq_len, self.np)
        
    def forward(self, x, l):
        out = self.yimgnet(x, l)
        out = self.lstmnet(out)

        return out
    
    def get_targets(self, targets):
        return coord_utils.locations_to_probability_maps(self.np.LocMapSize, targets)

    def get_location(self, pm):
        return coord_utils.probability_map_to_location(self.np.LocMapSize, pm)

    

