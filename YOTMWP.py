import torch
import torch.nn as nn
from torch.autograd import Variable
from YOTM import YOTM

class YOTMWP(YOTM):
    def __init__(self, batch_size, input_size, hidden_size, output_size, num_sequence, num_layers):
        super(YOTMWP, self).__init__()

        self.batch_size = batch_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = num_sequence
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.h_0 = self.init_hidden()
        
    def forward(self, x):
        x.view(self.batch_size, self.sequence_length, self.input_size)
        
        out, _ = self.lstm(x, self.h_0)
        out = self.fc(out)

        return out[:, self.sequence_length -1 :self.sequence_length, 1:5]
    
    def get_targets(self, targets):
        return targets
    
    def get_location(self, pm):
        return pm
