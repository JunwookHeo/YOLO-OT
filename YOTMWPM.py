# Class of model with Probability map.

from YOTM import *

class YOTMWPM(YOTM):
    def __init__(self):
        super(YOTMWPM, self).__init__()

    def get_loss_function(self):
        return nn.MSELoss(reduction='sum')
