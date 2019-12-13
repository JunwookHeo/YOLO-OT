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
    LayerSize = 3


class YimgNet(nn.Module):
    def __init__(self):
        super(YimgNet, self).__init__()

        self.conv1 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1)
        self.fc = nn.Linear(ModelParam.OutfiSize, ModelParam.OutCnnSize)

    def forward(self, x, l):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        out = torch.cat((x,l), 1)
        return out

class LstmNet(nn.Module):
    def __init__(self, batch_size, seq_len):
        super(LstmNet, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size=ModelParam.InLstmSize, hidden_size=ModelParam.HiddenSize, 
                            num_layers=ModelParam.LayerSize, batch_first=True)

        self.fc = nn.Linear(ModelParam.HiddenSize, ModelParam.LocSize)
        self.h_0 = self.init_hidden()

    def init_hidden(self):
            return (Variable(torch.zeros(ModelParam.LayerSize, self.batch_size, ModelParam.HiddenSize)), 
                Variable(torch.zeros(ModelParam.LayerSize, self.batch_size, ModelParam.HiddenSize)))

    def forward(self, x):
        x = x.view(x.size(0), self.seq_len, -1)
        out, _ = self.lstm(x, self.h_0)
        out = self.fc(out)
        return out


class YOTM(nn.Module):
    def __init__(self, batch_size, seq_len):
        super(YOTM, self).__init__()
        self.yimgnet = YimgNet()
        self.lstmnet = LstmNet(batch_size, seq_len)
     
    def forward(self, x, l):
        out = self.yimgnet(x, l)
        out = self.lstmnet(out)
        out = out.view(out.size(0), -1)

        return out
        
    def get_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=lr)
        return self.optimizer

    def get_loss_function(self):
        self.loss = torch.nn.MSELoss()
        return self.loss

