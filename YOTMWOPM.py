# Class of model without Probability map.

from YOTM import *

class YOTMWOPM(YOTM):
    def __init__(self):
        super(YOTMWOPM, self).__init__()

        self.loss_fn = nn.MSELoss(reduction='sum')

    def get_loss_function(self):
        return nn.MSELoss(reduction='sum')

