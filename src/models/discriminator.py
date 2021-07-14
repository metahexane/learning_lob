import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_seed = 54231856
torch.manual_seed(exp_seed)
np.random.seed(exp_seed)

"""
The discriminator used for all GANs
"""
class Discriminator(nn.Module):
 
    def __init__(self, in_dim, fm, out_sigmoid=False):
        super(Discriminator, self).__init__()

        self.out_sigmoid = out_sigmoid

        if self.out_sigmoid:
          self.main = nn.Sequential(
              # input is nc x 20 x 100
              nn.Conv2d(in_dim, fm, 2, 2, 0, bias=False),
              nn.BatchNorm2d(fm),
              nn.LeakyReLU(0.2),
  
              # state size is fm x 10 x 50
              nn.Conv2d(fm, fm * 2, (1, 2), (1, 2), 0, bias=False),
              nn.BatchNorm2d(fm * 2),
              nn.LeakyReLU(0.2),
  
              # state size is fm * 2 x 10 x 25
              nn.Conv2d(fm * 2, fm * 4, (1, 2), 1, 0, bias=False),
              nn.BatchNorm2d(fm * 4),
              nn.LeakyReLU(0.2),
  
              # state size is fm * 4 x 10 x 24
              nn.Conv2d(fm * 4, fm * 8, (10, 4), (10, 4), 0, bias=False),
              nn.BatchNorm2d(fm * 8),
              nn.LeakyReLU(0.2),
  
              # state size is fm * 8 x 1 x 6
          )
  
          self.linear = nn.Sequential(
              nn.Linear(fm * 8 * 6, fm * 8),
              nn.BatchNorm1d(fm * 8),
              nn.LeakyReLU(0.2),
  
              nn.Linear(fm * 8, fm),
              nn.BatchNorm1d(fm),
              nn.LeakyReLU(0.2),
  
              nn.Linear(fm, 1),
              nn.Sigmoid()
          )
        else:
          self.main = nn.Sequential(
              # input is nc x 20 x 100
              nn.Conv2d(in_dim, fm, 2, 2, 0, bias=False),
              nn.GroupNorm(1, fm),
              nn.LeakyReLU(0.2),
  
              # state size is fm x 10 x 50
              nn.Conv2d(fm, fm * 2, (1, 2), (1, 2), 0, bias=False),
              nn.GroupNorm(1, fm * 2),
              nn.LeakyReLU(0.2),
  
              # state size is fm * 2 x 10 x 25
              nn.Conv2d(fm * 2, fm * 4, (1, 2), 1, 0, bias=False),
              nn.GroupNorm(1, fm * 4),
              nn.LeakyReLU(0.2),
  
              # state size is fm * 4 x 10 x 24
              nn.Conv2d(fm * 4, fm * 8, (10, 4), (10, 4), 0, bias=False),
              nn.GroupNorm(1, fm * 8),
              # nn.LeakyReLU(0.2),

              nn.Sigmoid()
  
              # state size is fm * 8 x 1 x 6
          )
  
          self.linear = nn.Sequential(
              nn.Linear(fm * 8 * 6, fm * 8),
              nn.GroupNorm(1, fm * 8),
              nn.LeakyReLU(0.2),
  
              nn.Linear(fm * 8, fm),
              nn.GroupNorm(1, fm),
              nn.LeakyReLU(0.2),
  
              nn.Linear(fm, 1),
              nn.Sigmoid()
          )
 
    def forward(self, x):
        x_hat = self.main(x)
        x_lin = self.linear(x_hat.flatten(1))
        return x_lin