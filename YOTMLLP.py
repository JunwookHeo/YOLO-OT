import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
class LLPMParam:
    InfiSize = 128*52*52
    OutfiSize = 64*26*26
    OutCnnSize = 5 #8192
    LocSize = 5
    LocMapSize = 32*32
    InLstmSize = OutCnnSize + LocSize
    HiddenSize = 8 #4096
    LayerSize = 1
    OutputSize = 4

class LLNet(nn.Module):
    def __init__(self, batch_size, seq_len):
        super(LLNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.lstm1 = nn.LSTM(input_size=LLPMParam.InfiSize, hidden_size=LLPMParam.HiddenSize, 
                            num_layers=LLPMParam.LayerSize, batch_first=True)
        self.hidden1 = self.init_hidden()

        self.lstm2 = nn.LSTM(input_size=LLPMParam.LocSize, hidden_size=LLPMParam.HiddenSize, 
                            num_layers=LLPMParam.LayerSize, batch_first=True)
        self.hidden2 = self.init_hidden()

        self.fc = nn.Linear(LLPMParam.HiddenSize, LLPMParam.OutputSize)
        
    def init_hidden(self):
            return (Variable(torch.zeros(LLPMParam.LayerSize, self.batch_size, LLPMParam.HiddenSize)), 
                Variable(torch.zeros(LLPMParam.LayerSize, self.batch_size, LLPMParam.HiddenSize)))

    def forward(self, x, l):
        batch_size, seq_size, N, W, H = x.size()
        x = x.view(batch_size, seq_size, -1)
        c_x, _ = self.lstm1(x, self.hidden1)
        
        c_l, _ = self.lstm2(l, self.hidden2)
        c_out = c_x + c_l
        
        c_out = self.fc(c_out)
        return c_out


class YOTMLLP(nn.Module):
    def __init__(self, batch_size, seq_len):
        super(YOTMLLP, self).__init__()
        self.llnet = LLNet(batch_size, seq_len)
     
    def forward(self, x, l):
        out = self.llnet(x, l)
        out = out.view(out.size(0), out.size(1), -1)

        return out
        
    def get_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=lr)
        return self.optimizer

    def get_loss_function(self):
        self.loss = torch.nn.MSELoss()
        return self.loss

