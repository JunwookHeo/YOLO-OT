import torch
import torch.nn as nn
from torch.autograd import Variable

from coord_utils import *

from YOTMWPM import *

class LstmNet_PM(nn.Module):
    def __init__(self, device, batch_size, seq_len, np):
        super(LstmNet_PM, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.np = np

        self.lstm = nn.LSTM(input_size=self.np.InLstmSize, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, batch_first=True)

        #self.fc = nn.Linear(self.np.HiddenSize, self.np.OutputSize)
        
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize).to(self.device)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize).to(self.device)))

    def forward(self, x):
        batch_size, seq_size, N = x.size()
        x = x.view(batch_size*seq_size, N)
        x = coord_utils.locations_to_probability_maps(self.np.LocMapSize, x)
        x = x.view(batch_size, seq_size, -1)

        c_out, _ = self.lstm(x, self.hidden)
        return c_out

class YOTMPMO(YOTMWPM):
    class NP:
        LocSize = 5
        LocMapSize = 32
        InLstmSize = LocMapSize*LocMapSize
        HiddenSize = 1024 
        LayerSize = 1
        OutputSize = LocMapSize*LocMapSize
        
    def __init__(self, batch_size, seq_len):
        super(YOTMPMO, self).__init__()
        self.np = self.NP()

        self.lstmnet = LstmNet_PM(self.device, batch_size, seq_len, self.np)
        
    def forward(self, x, l):
        out = self.lstmnet(l)

        return out
    
    def get_targets(self, targets):
        return coord_utils.locations_to_probability_maps(self.np.LocMapSize, targets)

    def get_location(self, pm):
        return coord_utils.probability_map_to_location(self.np.LocMapSize, pm)

    

