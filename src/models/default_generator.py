import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_seed = 54231856
torch.manual_seed(exp_seed)
np.random.seed(exp_seed)

"""
The GAN used for generation/prediction. This is the same for DCGAN and WGAN.
"""
class DefaultGenerator(nn.Module):
    def __init__(self, in_dim, noise_dim, fm, prediction_mode=False):
        super(DefaultGenerator, self).__init__()

        if prediction_mode:
          self.main = nn.Sequential(
  
              # input is 1 x 20 x 70
              nn.Conv2d(1, fm, 2, 2, 0, bias=False),
              nn.BatchNorm2d(fm),
              nn.LeakyReLU(0.2),

              # input is fm x 10 x 35
              nn.Conv2d(fm, fm * 2, (1,2), (1,2), 0, bias=False),
              nn.BatchNorm2d(fm * 2),
              nn.LeakyReLU(0.2),

              # input is fm x 10 x 17
              nn.Conv2d(fm * 2, fm * 4, (3,2), (1,2), 0, bias=False),
              nn.BatchNorm2d(fm * 4),
              nn.LeakyReLU(0.2),
  
              # input is fm x 8 x 8
              nn.ConvTranspose2d(fm * 4, fm * 4, 4, 2, 1, bias=False),
              nn.BatchNorm2d(fm * 4),
              nn.LeakyReLU(0.2),

              # input is fm x 16 x 16
              nn.ConvTranspose2d(fm * 4, fm * 2, (5,4), (1,2), (0,1), bias=False),
              nn.BatchNorm2d(fm * 2),
              nn.LeakyReLU(0.2),

              # input is fm x 20 x 32
              nn.Conv2d(fm * 2, fm, (1, 3), 1, 0, bias=False),
              nn.BatchNorm2d(fm),
              nn.LeakyReLU(0.2),
  
              # input is fm x 20 x 30
              nn.Conv2d(fm, 1, 1, 1, 0),
              nn.Sigmoid(),
  
              # output is 1 x 20 x 30
          )
        else:
          self.main = nn.Sequential(
  
              # input is 100 x 1 x 1
              nn.ConvTranspose2d(noise_dim, fm * 8, 2, 1, 0, bias=False),
              nn.BatchNorm2d(fm * 8),
              nn.LeakyReLU(0.2),
  
              # input is fm * 8 x 2 x 2
              nn.ConvTranspose2d(fm * 8, fm * 4, 4, 2, 1, bias=False),
              nn.BatchNorm2d(fm * 4),
              nn.LeakyReLU(0.2),
  
              # input is fm * 2 x 4 x 4
              nn.ConvTranspose2d(fm * 4, fm * 2, 4, 2, 1, bias=False),
              nn.BatchNorm2d(fm * 2),
              nn.LeakyReLU(0.2),
  
              # input is fm x 8 x 8
              nn.ConvTranspose2d(fm * 2, fm, 4, 2, 1, bias=False),
              nn.BatchNorm2d(fm),
              nn.LeakyReLU(0.2),

              # input is fm x 16 x 16
              nn.ConvTranspose2d(fm, int(fm / 2), (5,4), (1,2), (0,1), bias=False),
              nn.BatchNorm2d(int(fm / 2)),
              nn.LeakyReLU(0.2),

              # input is fm x 20 x 32
              nn.ConvTranspose2d(int(fm / 2), int(fm / 4), (1,4), (1,2), (0,1), bias=False),
              nn.BatchNorm2d(int(fm / 4)),
              nn.LeakyReLU(0.2),
  
              # input is fm x 20 x 64
              nn.ConvTranspose2d(int(fm / 4), 1, (1,2), (1,2), (0,14)),
              nn.Sigmoid(),
  
              # output is 1 x 20 x 100
          )
      
    def forward(self, x, spreads=False, debug=False, fts_only=False):
        return self.main(x)