import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_seed = 54231856
torch.manual_seed(exp_seed)
np.random.seed(exp_seed)

"""
The CNN used for prediction
"""
class CNNPred(nn.Module):
    def __init__(self, fm):
        super(CNNPred, self).__init__()

        self.main = nn.Sequential(

            # input is 1 x 20 x 70
            nn.Conv2d(1, fm, (1, 11), 1, 0, bias=False),
            nn.BatchNorm2d(fm),
            nn.LeakyReLU(0.2),

            # input is fm x 20 x 60
            nn.Conv2d(fm, fm * 2, (1,11), 1, 0, bias=False),
            nn.BatchNorm2d(fm * 2),
            nn.LeakyReLU(0.2),

            # input is fm x 20 x 50
            nn.Conv2d(fm * 2, fm * 4, (1,11), 1, 0, bias=False),
            nn.BatchNorm2d(fm * 4),
            nn.LeakyReLU(0.2),

            # input is fm x 20 x 40
            nn.Conv2d(fm * 4, fm * 8, (1,6), 1, 0, bias=False),
            nn.BatchNorm2d(fm * 8),
            nn.LeakyReLU(0.2),

            # input is fm x 20 x 35
            nn.Conv2d(fm * 8, 1, (1,6), 1, 0),
            nn.Sigmoid(),

            # output is 1 x 20 x 30
        )

    def forward(self, x):
      return self.main(x)