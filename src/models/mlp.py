import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_seed = 54231856
torch.manual_seed(exp_seed)
np.random.seed(exp_seed)

"""
The MLP used for prediction
"""
class MLP(nn.Module):
 
    def __init__(self, L=10, T_base=35, T_pred=15):
      super(MLP, self).__init__()
      
      self.L = L
      self.T_base = T_base
      self.T_pred = T_pred

      in_dim = int(L * T_base * 4)
      out_dim = int(L * T_pred * 4)
      # In goes 10 x 35 x 4 (=350), out goes 10 x 15 (=150)
      self.main = nn.Sequential(
          nn.Linear(in_dim, 512),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(.2),

          nn.Linear(512, 256),
          nn.BatchNorm1d(256),
          nn.LeakyReLU(.2),

          nn.Linear(256, 128),
          nn.BatchNorm1d(128),
          nn.LeakyReLU(.2),

          nn.Linear(128, 64),
          nn.BatchNorm1d(64),
          nn.LeakyReLU(.2),

          nn.Linear(64, out_dim),
          nn.Sigmoid()
      )

    def forward(self, x):
      out = self.main(x)
      return out