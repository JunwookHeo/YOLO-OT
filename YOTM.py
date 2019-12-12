import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1)
        self.fc = nn.Linear(64*26*26, 8192)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*26*26)
        return self.fc(x)

class YOTM(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, output_size, num_sequence, num_layers):
        super(YOTM, self).__init__()

        self.batch_size = batch_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = num_sequence
        self.output_size = output_size
        self.cnn = CNN()
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True)


        self.fco = nn.Linear(4016, 5)

        #self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.h_0 = self.init_hidden()
        
    def forward(self, x):
        c_in = x.view(self.batch_size, self.sequence_length, self.input_size)
        c_out = self.cnn(c_in)
        c_in = x.view(self.batch_size, self.sequence_length, self.input_size)
        c_out, _ = self.lstm(c_in, self.h_0)

        out = self.fc(c_out)

        return out[:, self.sequence_length -1 :self.sequence_length, 1:5]
        
    def init_hidden(self):
            return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)),
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))

    def get_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=lr)
        return self.optimizer

    def get_loss_function(self):
        self.loss = torch.nn.MSELoss()
        return self.loss

