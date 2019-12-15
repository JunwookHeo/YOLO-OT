import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
class ModelParam:
    InfiSize = 128*52*52
    OutfiSize = 64*26*26
    OutCnnSize = 8192
    LocSize = 5
    LocMapSize = 32*32
    InLstmSize = OutCnnSize + LocSize
    HiddenSize = 4096
    LayerSize = 1
    OutputSize = 4


class YimgNet(nn.Module):
    def __init__(self):
        super(YimgNet, self).__init__()

        self.conv1 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1)
        self.fc = nn.Linear(ModelParam.OutfiSize, ModelParam.OutCnnSize)

    def forward(self, x, l):
        batch_size, seq_size, C, H, W = x.size()
        c_in = x.view(batch_size*seq_size, C, H, W)
        c_in = F.relu(self.conv1(c_in))
        c_in = F.relu(self.conv2(c_in))
        c_in = F.relu(self.conv3(c_in))
        c_in = c_in.view(batch_size*seq_size, -1)
        c_in = self.fc(c_in)
        c_out = c_in.view(batch_size, seq_size, -1)
        out = torch.cat((c_out,l), 2)
        return out

class LstmNet(nn.Module):
    def __init__(self, batch_size, seq_len):
        super(LstmNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size=ModelParam.InLstmSize, hidden_size=ModelParam.HiddenSize, 
                            num_layers=ModelParam.LayerSize, batch_first=True)

        self.fc = nn.Linear(ModelParam.HiddenSize, ModelParam.OutputSize)
        self.h_0 = self.init_hidden()

    def init_hidden(self):
            return (Variable(torch.zeros(ModelParam.LayerSize, self.batch_size, ModelParam.HiddenSize)), 
                Variable(torch.zeros(ModelParam.LayerSize, self.batch_size, ModelParam.HiddenSize)))

    def forward(self, x):
        x = x.view(x.size(0), self.seq_len, -1)
        c_out, _ = self.lstm(x, self.h_0)
        c_out = self.fc(c_out)
        return c_out


class YOTM(nn.Module):
    def __init__(self, batch_size, seq_len):
        super(YOTM, self).__init__()
        self.yimgnet = YimgNet()
        self.lstmnet = LstmNet(batch_size, seq_len)
     
    def forward(self, x, l):
        out = self.yimgnet(x, l)
        out = self.lstmnet(out)
        out = out.view(out.size(0), out.size(1), -1)

        return out
        
    def get_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=lr)
        return self.optimizer

    def get_loss_function(self):
        self.loss = torch.nn.MSELoss()
        return self.loss

