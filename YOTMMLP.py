import torch
import torch.nn as nn
from torch.autograd import Variable

from YOTM import *

class MlpLNet(nn.Module):
    def __init__(self, batch_size, seq_len, np):
        super(MlpLNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.np = np

        self.lstmx = nn.LSTM(input_size=2, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, dropout=0.3, batch_first=True)
        self.lstmy = nn.LSTM(input_size=2, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, dropout=0.3, batch_first=True)
        self.lstmw = nn.LSTM(input_size=2, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, dropout=0.3, batch_first=True)
        self.lstmh = nn.LSTM(input_size=2, hidden_size=self.np.HiddenSize, 
                            num_layers=self.np.LayerSize, dropout=0.3, batch_first=True)
        
        self.init_hidden()

        self.fcx = nn.Linear(self.np.HiddenSize, 1)
        self.fcy = nn.Linear(self.np.HiddenSize, 1)
        self.fcw = nn.Linear(self.np.HiddenSize, 1)
        self.fch = nn.Linear(self.np.HiddenSize, 1)
        
        
    def init_hidden(self):
        self.hiddenx = (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)))
        self.hiddeny = (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)))
        self.hiddenw = (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)))
        self.hiddenh = (Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)), 
                Variable(torch.zeros(self.np.LayerSize, self.batch_size, self.np.HiddenSize)))


    def forward(self, x, l):
        batch_size, seq_size, N = l.size()
        l = l.view(batch_size, seq_size, -1)
        
        x_out = l[:,:,0].view(batch_size, seq_size, -1)
        y_out = l[:,:,1].view(batch_size, seq_size, -1)
        w_out = l[:,:,2].view(batch_size, seq_size, -1)
        h_out = l[:,:,3].view(batch_size, seq_size, -1)
        p_out = l[:,:,4].view(batch_size, seq_size, -1)

        x_out, _ = self.lstmx(torch.cat((x_out, p_out), 2), self.hiddenx)
        y_out, _ = self.lstmy(torch.cat((y_out, p_out), 2), self.hiddeny)
        w_out, _ = self.lstmw(torch.cat((w_out, p_out), 2), self.hiddenw)
        h_out, _ = self.lstmh(torch.cat((h_out, p_out), 2), self.hiddenh)
        
        x_out = self.fcx(x_out)
        y_out = self.fcy(y_out)
        w_out = self.fcw(w_out)
        h_out = self.fch(h_out)

        c_out = torch.cat((x_out, y_out, w_out, h_out), 2)
        
        return c_out


class YOTMMLP(YOTM):
    class NP:
        LocSize = 5
        LocMapSize = 32*32
        HiddenSize = 32
        LayerSize = 1
        OutputSize = 4
        
    def __init__(self, batch_size, seq_len):
        super(YOTMMLP, self).__init__()
        self.np = self.NP()

        self.mlplnet = MlpLNet(batch_size, seq_len, self.np)
     
    def forward(self, x, l):
        batch_size, seq_size, _ = l.size()        
        out = self.mlplnet(x, l)
        return out
    
    def init_hidden(self):
        self.lstmlnet.init_hidden()
    