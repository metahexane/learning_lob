import torch
import numpy as np
import torch.nn as nn
from torch.distributions.uniform import Uniform
import math
from scipy import special
from torchvectorized.vlinalg import vSymEig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_seed = 54231856
torch.manual_seed(exp_seed)
np.random.seed(exp_seed)

"""
The whole model, including the generator and the mathematical model
"""
class Generator(nn.Module):
 
    def __init__(self, in_dim, noise_dim, fm, S=10, L=10, T=50, gamma=0.90, sampling_rate=5, prediction_mode=False, implied_T=35):
        super(Generator, self).__init__()
 
        self.S = S
        self.L = L
        self.T = T
        self.gamma = gamma
        self.sampling_rate = sampling_rate
        self.implied_T = implied_T
 
        unif_sampler = Uniform(torch.Tensor([-4.0]), torch.Tensor([4.0]))
        self.prediction_mode = prediction_mode
        
        ranget_col = torch.arange(0, T, device=device)
        ranget_row = torch.arange(0, L, device=device)
 
        ranget_level = torch.arange(0, L - 1, device=device)
        ranget_next_level = ranget_level + 1
 
        self.ranget_col_all = ranget_col.repeat_interleave(len(ranget_level))
        self.ranget_level_all = ranget_level.repeat(len(ranget_col))
        self.ranget_next_level_all = ranget_next_level.repeat(len(ranget_col))
 
        self.even_indices_col = 2 * ranget_col
        self.odd_indices_col = self.even_indices_col + 1
        self.even_indices_col = self.even_indices_col.repeat(len(ranget_row))
        self.odd_indices_col = self.odd_indices_col.repeat(len(ranget_row))
        self.even_indices_row = 2 * ranget_row
        self.odd_indices_row = self.even_indices_row + 1
        self.even_indices_row = self.even_indices_row.repeat_interleave(len(ranget_col))
        self.odd_indices_row = self.odd_indices_row.repeat_interleave(len(ranget_col))
        
        # L x (T - 1) (L = 10, T = 50)
 
        # Initialize model matrices --------------------------------------------
        self.pa_transition_weights = nn.Parameter(unif_sampler.sample((T, S, L, 1, 2)).squeeze(5).to(device))
        self.pb_transition_weights = nn.Parameter(unif_sampler.sample((T, S, L, 1, 2)).squeeze(5).to(device))
        self.va_transition_weights = nn.Parameter(unif_sampler.sample((T, S, L, 1, 2)).squeeze(5).to(device))
        self.vb_transition_weights = nn.Parameter(unif_sampler.sample((T, S, L, 1, 2)).squeeze(5).to(device))
 
        self.pa_transition_probs = nn.Parameter(unif_sampler.sample((S, L, 3, 3)).squeeze(4).to(device))
        self.pb_transition_probs = nn.Parameter(unif_sampler.sample((S, L, 3, 3)).squeeze(4).to(device))
        self.va_transition_probs = nn.Parameter(unif_sampler.sample((S, L, 3, 3)).squeeze(4).to(device))
        self.vb_transition_probs = nn.Parameter(unif_sampler.sample((S, L, 3, 3)).squeeze(4).to(device))
        
        self.sampling_weights = torch.eye(self.L, device=device)
        self.sampling_weights = self.sampling_weights.repeat_interleave(self.L, 1)
        self.sampling_weights = (self.L - 1) * self.sampling_rate * self.sampling_weights + 1

        if prediction_mode:
          nn.init.kaiming_uniform_(self.pa_transition_weights)
          nn.init.kaiming_uniform_(self.pb_transition_weights)
          nn.init.kaiming_uniform_(self.va_transition_weights)
          nn.init.kaiming_uniform_(self.vb_transition_weights)
        # ----------------------------------------------------------------------
 
        # Initialize base ------------------------------------------------------
        if prediction_mode:
          self.main = nn.Sequential(
              # input is 1 x 2L x 2T
              # input is 1 x 20 x 70
              nn.Conv2d(1, fm, 2, 2, 0, bias=False),
              nn.BatchNorm2d(fm),
              nn.LeakyReLU(0.2),

              # input is fm x 10 x 35
              nn.Conv2d(fm, fm*2, (1,2), (1,2), 0, bias=False),
              nn.BatchNorm2d(fm*2),
              nn.LeakyReLU(0.2),

              # input is fm x 10 x 17
              nn.Conv2d(fm*2, fm*4, (3,2), (1,2), 0, bias=False),
              nn.BatchNorm2d(fm*4),
              nn.LeakyReLU(0.2),

              # output is fm*4 x 8 x 8 
          )
          last_output_fts = fm * 4
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
  
              # output is fm * 2 x 8 x 8
          )
          last_output_fts = fm * 2
        # ----------------------------------------------------------------------
 
        # CONSTRUCT INIT VALUES LAYERS -----------------------------------------
 
        fix_stride = 1
        fix_padding = 0
        output_ks_init = int(L - 7 * fix_stride + 2 * fix_padding)
 
        if not self.prediction_mode:
          self.init_values = nn.Sequential(
              # input is fm * 2 x 8 x 8
              nn.ConvTranspose2d(last_output_fts, fm, (output_ks_init, 3), (fix_stride, 1), (fix_padding,3), bias=False),
              nn.BatchNorm2d(fm),
              nn.LeakyReLU(0.2),
  
              nn.Conv2d(fm, int(fm / 2), 1, 1, 0, bias=False),
              nn.BatchNorm2d(int(fm / 2)),
              nn.LeakyReLU(0.2),
  
              nn.Conv2d(int(fm / 2), int(fm / 4), 1, 1, 0, bias=False),
              nn.BatchNorm2d(int(fm / 4)),
              nn.LeakyReLU(0.2),
  
              nn.Conv2d(int(fm / 4), 1, 1, 1, 0),
              nn.Sigmoid(),
  
              # output is L x 4
          )
        # ----------------------------------------------------------------------

        # CONSTRUCT RATIO MAPPING LAYER ----------------------------------------
        # INPUT: [1024, 1, L, IMPLIED_T - 1] (34)
        # OUTPUT: [1024, 1, L, T] (15)
        # if self.prediction_mode:
        self.pa_ratio_up_layer = self.construct_ratio_layer()
        self.pa_ratio_down_layer = self.construct_ratio_layer()
        self.pb_ratio_up_layer = self.construct_ratio_layer()
        self.pb_ratio_down_layer = self.construct_ratio_layer()
        self.va_ratio_up_layer = self.construct_ratio_layer()
        self.va_ratio_down_layer = self.construct_ratio_layer()
        self.vb_ratio_up_layer = self.construct_ratio_layer()
        self.vb_ratio_down_layer = self.construct_ratio_layer()
 
        # CONSTRUCT S LAYERS ---------------------------------------------------
        layer_amount = math.floor(math.log2(S)) - 3 
        # S has a minimum value of 8
        self.S_values = [
          # input is fm * 2 x 8 x 8
          nn.ConvTranspose2d(last_output_fts, fm, (1, 5), 1, 0, bias=False),
          nn.BatchNorm2d(fm),
          nn.LeakyReLU(0.2)
          # output is fm x 8 x 12
        ]
 
        for i in range(layer_amount):
          self.S_values.append(nn.ConvTranspose2d(fm , fm, (4, 1), (2, 1), (1, 0), bias=False))
          self.S_values.append(nn.BatchNorm2d(fm))
          self.S_values.append(nn.LeakyReLU(0.2))
        
        covered_layers = 2 ** (layer_amount + 3)
        missing_layers = S - covered_layers
        
        if missing_layers > 0:
          output_ks_s = int(S - (covered_layers - 1) * fix_stride + 2 * fix_padding)
          self.S_values.append(nn.ConvTranspose2d(fm, fm, (output_ks_s, 1), (fix_stride, 1), (fix_padding, 0), bias=False))
          self.S_values.append(nn.BatchNorm2d(fm))
          self.S_values.append(nn.LeakyReLU(0.2))
 
        self.S_values.append(nn.Conv2d(fm, int(fm / 2), 1, 1, 0, bias=False))
        self.S_values.append(nn.BatchNorm2d(int(fm / 2)))
        self.S_values.append(nn.LeakyReLU(0.2))
 
        self.S_values.append(nn.Conv2d(int(fm / 2), int(fm / 4), 1, 1, 0, bias=False))
        self.S_values.append(nn.BatchNorm2d(int(fm / 4)))
        self.S_values.append(nn.LeakyReLU(0.2))
        
        self.S_values.append(nn.Conv2d(int(fm / 4), 1, 1, 1, 0))
        self.S_values.append(nn.BatchNorm2d(1))
        self.S_values.append(nn.LeakyReLU(0.2))
        
        self.S_values = nn.Sequential(*self.S_values)
        # ----------------------------------------------------------------------
 
        # Initialize layer weights ---------------------------------------------
        nn.init.kaiming_normal_(self.main[0].weight)
        nn.init.kaiming_normal_(self.main[3].weight)
        nn.init.kaiming_normal_(self.main[6].weight)

        if self.prediction_mode:
          nn.init.kaiming_normal_(self.pa_ratio_up_layer[0].weight)
          nn.init.kaiming_normal_(self.pa_ratio_down_layer[0].weight)
          nn.init.kaiming_normal_(self.pb_ratio_up_layer[0].weight)
          nn.init.kaiming_normal_(self.pb_ratio_down_layer[0].weight)
          nn.init.kaiming_normal_(self.va_ratio_up_layer[0].weight)
          nn.init.kaiming_normal_(self.va_ratio_down_layer[0].weight)
          nn.init.kaiming_normal_(self.vb_ratio_up_layer[0].weight)
          nn.init.kaiming_normal_(self.vb_ratio_down_layer[0].weight)
        else:
          nn.init.kaiming_normal_(self.init_values[0].weight)
          nn.init.kaiming_normal_(self.init_values[3].weight)
          nn.init.kaiming_normal_(self.init_values[6].weight)
          nn.init.kaiming_normal_(self.init_values[9].weight)
 
        for i in range(len(self.S_values)):
          if i % 3 == 0:
            nn.init.kaiming_normal_(self.S_values[i].weight)
        # ----------------------------------------------------------------------
 
        self.sigmoid = nn.Sigmoid()

    def construct_ratio_layer(self):
      """
      Constructs the layer that takes care of the mapping for prediction (ConvNet)
      """
      return nn.Sequential(
          # input is 1 x 10 x 34
          nn.Conv2d(1, 1, (1, 5), (1,2), 0),
          nn.BatchNorm2d(1),
          nn.LeakyReLU(.2)
      )
 
    def get_stat(self, matr):
        """
        Calculate the stationary distributions of input matrices
        """
        sym = .5 * (matr.transpose(2, 3) + matr)
        re = sym.reshape(sym.shape[0], sym.shape[1], 9).transpose(1, 2).unsqueeze(3).unsqueeze(4) + 1e-10
        _, eig_vecs = vSymEig(re, eigenvectors=True)
        eig_vecs = eig_vecs.transpose(1,3).transpose(2,3).squeeze(5).squeeze(4)
        prod = eig_vecs[:, :, :, -1].unsqueeze(2) @ sym
        return (prod / prod.sum(3, keepdim=True))
 
    def ext_sigmoid(self, x, k=1, scale=1):
        """
        The extended logistic sigmoid function
        """
        return 1 / (1 + torch.exp(- k * x)) / scale
 
    def update_order_matrix(self, 
                            x_0: torch.Tensor, 
                            lo_weights: nn.Parameter,
                            nn_probs: nn.Parameter,
                            inc_selection: torch.Tensor,
                            dec_selection: torch.Tensor,
                            prob_selection: torch.Tensor,
                            debug=False):
        """
        The mathematical model
        """
        
        bs = x_0.shape[0]
 
        x_0_diag = torch.diag_embed(x_0)
 
        if not self.prediction_mode:
          nn_probs_softmax = torch.softmax(nn_probs, dim=3) # dim: 10 x 10 x 3 x 3
          nn_probs_softmax = nn_probs_softmax.repeat((bs, 1, 1, 1, 1)) # bs x 10 x 10 x 3 x 3
        else:
          nn_probs_softmax = nn_probs

        # Acquire weighted transition matrices
        p_weight_probs = torch.softmax(prob_selection.reshape((bs, self.S, 1, 1, 1)), dim=1) # bs x 10 x 1 x 1 x 1
        nn_probs_softmax = (p_weight_probs * nn_probs_softmax).sum(dim=1) # bs x 10 x 3 x 3 
 
        # Instead of random shuffling, have 10x T_1, 10x T_2, etc.
        a_rep = nn_probs_softmax.repeat_interleave(self.L, 1)
        a_shuff = nn_probs_softmax.repeat((1, self.L, 1, 1)) # a_rep[:, torch.randperm(100), : , :]
        a_prods = a_rep @ a_shuff
 
        # Acquire sampling set
        pos_vals = self.get_stat(a_prods) # dim: [B x L^2 x 1 x 3]
 
        # Static weights
        # When shuffling, place more weight on first 10 stationary distributions for L1, then next 10 for L2, etc.
        w = torch.multinomial(self.sampling_weights, self.T, replacement=True)
        vals = pos_vals[:, w, :, :] # dim: [B, L, T, 1, 3]
        vals = vals.transpose(1, 2)
        mval = vals.mean(dim=4, keepdim=True)
        exp_weights = (1 - self.gamma ** (torch.arange(0, self.T, device=device))).unsqueeze(1).repeat((bs, 1, self.L)).unsqueeze(3).unsqueeze(4)
        p_probs = mval + (vals - mval) * exp_weights
 
        # sizes are 10 x 10 x 49 ( batch x L x L x (T - 1) )
        if not self.prediction_mode:
          up_ratios = 1 + self.ext_sigmoid(lo_weights[:, :, :, :, 0], 1, 50) # 50 x 10 x 10 x 1
          down_ratios = 1 - self.ext_sigmoid(lo_weights[:, :, :, :, 1], 1, 50)
        
          up_ratios = up_ratios.repeat((bs, 1, 1, 1, 1)) # bs x 50 x 10 x 10 x 1
          dec_ratios = down_ratios.repeat((bs, 1, 1, 1, 1)) # bs x 50 x 10 x 10 x 1
        else:
          up_ratios = 1 + self.ext_sigmoid(lo_weights[:, :, :, :, :, 0], 1, 1)
          dec_ratios = 1 - self.ext_sigmoid(lo_weights[:, :, :, :, :, 1], 1, 1)
        
        # Acquire weighted rate matrices
        up_selection_r = torch.softmax(inc_selection.reshape((bs, 1, self.S, 1, 1)), dim=2) # bs x 1 x 10 x 1 x 1
        up_ratios = (up_selection_r * up_ratios).sum(dim=2) # bs x 50 x 10 x 1
 
        dec_selection_r = torch.softmax(dec_selection.reshape((bs, 1, self.S, 1, 1)), dim=2) # bs x 1 x 10 x 1 x 1
        dec_ratios = (dec_selection_r * dec_ratios).sum(dim=2) # bs x 50 x 10 x 1
 
        stat_ratios = torch.ones((bs, self.T, self.L, 1), device=device)
        ratio_matrix = torch.cat((up_ratios, stat_ratios, dec_ratios), dim=3).unsqueeze(3) # dim: bs x 50 x 10 x 1 x 3
        
        # Calculate expectation
        weighted_ratios = (p_probs * ratio_matrix).sum(dim=4) # dim is now bs x 49 x 10 x 1 
        weighted_ratios = weighted_ratios.transpose(1, 2).squeeze(3)
        weighted_ratios = weighted_ratios.cumprod(dim=2)
 
        # out shape: B x 10 x 10 x 50
        # Compute values E[y_0 | x_0], E[y_1 | y_0], E[y_2 | y_1], etc.
        layer_out = torch.bmm(x_0_diag.squeeze(1), weighted_ratios).unsqueeze(1)
        return layer_out, p_probs
 
    def construct_order_book_matrix(self, pa, pb, va, vb):
        """
        Constructs the order book matrix given the values of p_a, p_b, v_a, v_b.
        """
        bs = pa.shape[0]
 
        total_L = 2 * self.L
        total_T = 2 * self.T
        total_dim = self.L * self.T
 
        orderbook = torch.empty((bs, 1, total_L, total_T), device=device)
 
        orderbook[:, :, self.even_indices_row, self.even_indices_col] = pa.reshape(bs, 1, total_dim)
        orderbook[:, :, self.odd_indices_row, self.even_indices_col] = pb.reshape(bs, 1, total_dim)
        orderbook[:, :, self.even_indices_row, self.odd_indices_col] = va.reshape(bs, 1, total_dim)
        orderbook[:, :, self.odd_indices_row, self.odd_indices_col] = vb.reshape(bs, 1, total_dim)
 
        return orderbook

    def get_transition_matrix(self, x):
      """
      Acquires the transition matrices for input data x
      """
      # Input is B x 1 x L x T
      pre_indices = torch.arange(0, self.implied_T - 2)
      post_indices = torch.arange(1, self.implied_T - 1)
      post_2nd_indices = torch.arange(2, self.implied_T)
      
      down_down = (x[:, :, :, post_indices] < x[:, :, :, pre_indices]) * (x[:, :, :, post_2nd_indices] < x[:, :, :, post_indices])
      down_stag = (x[:, :, :, post_indices] < x[:, :, :, pre_indices]) * (x[:, :, :, post_2nd_indices] == x[:, :, :, post_indices])
      down_up   = (x[:, :, :, post_indices] < x[:, :, :, pre_indices]) * (x[:, :, :, post_2nd_indices] > x[:, :, :, post_indices])

      stag_down = (x[:, :, :, post_indices] == x[:, :, :, pre_indices]) * (x[:, :, :, post_2nd_indices] < x[:, :, :, post_indices])
      stag_stag = (x[:, :, :, post_indices] == x[:, :, :, pre_indices]) * (x[:, :, :, post_2nd_indices] == x[:, :, :, post_indices])
      stag_up   = (x[:, :, :, post_indices] == x[:, :, :, pre_indices]) * (x[:, :, :, post_2nd_indices] > x[:, :, :, post_indices])

      up_down   = (x[:, :, :, post_indices] > x[:, :, :, pre_indices]) * (x[:, :, :, post_2nd_indices] < x[:, :, :, post_indices])
      up_stag   = (x[:, :, :, post_indices] > x[:, :, :, pre_indices]) * (x[:, :, :, post_2nd_indices] == x[:, :, :, post_indices])
      up_up     = (x[:, :, :, post_indices] > x[:, :, :, pre_indices]) * (x[:, :, :, post_2nd_indices] > x[:, :, :, post_indices])

      transition_matrices = torch.empty((x.shape[0], self.L, 3, 3), device=device)
      transition_matrices[:, :, 2, 0] = torch.sum(down_down, dim=3).squeeze(1) / (self.implied_T - 2)
      transition_matrices[:, :, 2, 1] = torch.sum(down_stag, dim=3).squeeze(1) / (self.implied_T - 2)
      transition_matrices[:, :, 2, 2] = torch.sum(down_up, dim=3).squeeze(1) / (self.implied_T - 2)

      transition_matrices[:, :, 1, 0] = torch.sum(stag_down, dim=3).squeeze(1) / (self.implied_T - 2)
      transition_matrices[:, :, 1, 1] = torch.sum(stag_stag, dim=3).squeeze(1) / (self.implied_T - 2)
      transition_matrices[:, :, 1, 2] = torch.sum(stag_up, dim=3).squeeze(1) / (self.implied_T - 2)
      
      transition_matrices[:, :, 0, 0] = torch.sum(up_down, dim=3).squeeze(1) / (self.implied_T - 2)
      transition_matrices[:, :, 0, 1] = torch.sum(up_stag, dim=3).squeeze(1) / (self.implied_T - 2)
      transition_matrices[:, :, 0, 2] = torch.sum(up_up, dim=3).squeeze(1) / (self.implied_T - 2)

      return transition_matrices

    def extract_values(self, x):
      """
      Extract rates and transition matrices from input data
      """

      # x dim: B x 1 x 2L x 2T
      ask_indices = torch.arange(0, self.L) * 2
      bid_indices = torch.arange(0, self.L) * 2 + 1

      # Initial values
      pa_init = x[:, :, ask_indices, -2]
      pb_init = x[:, :, bid_indices, -2]
      va_init = x[:, :, ask_indices, -1]
      vb_init = x[:, :, bid_indices, -1]

      # Extract all values
      ranget_col = torch.arange(0, self.implied_T, device=device)
      even_indices_col = 2 * ranget_col
      odd_indices_col = even_indices_col + 1
      even_indices_col = even_indices_col.repeat(self.L)
      odd_indices_col = odd_indices_col.repeat(self.L)

      ranget_row = torch.arange(0, self.L, device=device)
      even_indices_row = 2 * ranget_row
      odd_indices_row = even_indices_row + 1
      even_indices_row = even_indices_row.repeat_interleave(self.implied_T)
      odd_indices_row = odd_indices_row.repeat_interleave(self.implied_T)

      pa = x[:, :, even_indices_row, even_indices_col].reshape((x.shape[0], 1, self.L, self.implied_T))
      pb = x[:, :, odd_indices_row, even_indices_col].reshape((x.shape[0], 1, self.L, self.implied_T))
      va = x[:, :, even_indices_row, odd_indices_col].reshape((x.shape[0], 1, self.L, self.implied_T))
      vb = x[:, :, odd_indices_row, odd_indices_col].reshape((x.shape[0], 1, self.L, self.implied_T))

      # Ratios
      pre_indices = torch.arange(0, self.implied_T - 1)
      post_indices = torch.arange(1, self.implied_T)
      eps = 1e-6

      pa_ratios = (pa[:, :, :, post_indices] + eps) / (pa[:, :, :, pre_indices] + eps)
      pb_ratios = (pb[:, :, :, post_indices] + eps) / (pb[:, :, :, pre_indices] + eps)
      va_ratios = (va[:, :, :, post_indices] + eps) / (va[:, :, :, pre_indices] + eps)
      vb_ratios = (vb[:, :, :, post_indices] + eps) / (vb[:, :, :, pre_indices] + eps)

      pa_ratios_up_c = pa_ratios.clone()
      pa_ratios_down_c = pa_ratios.clone()
      pa_ratios_up_c[pa_ratios_up_c < 1] = 2 - pa_ratios_up_c[pa_ratios_up_c < 1]
      pa_ratios_down_c[pa_ratios_down_c > 1] = 2 - pa_ratios_down_c[pa_ratios_down_c > 1]

      pb_ratios_up_c = pb_ratios.clone()
      pb_ratios_down_c = pa_ratios.clone()
      pb_ratios_up_c[pb_ratios_up_c < 1] = 2 - pb_ratios_up_c[pb_ratios_up_c < 1]
      pb_ratios_down_c[pb_ratios_down_c > 1] = 2 - pb_ratios_down_c[pb_ratios_down_c > 1]

      va_ratios_up_c = va_ratios.clone()
      va_ratios_down_c = va_ratios.clone()
      va_ratios_up_c[va_ratios_up_c < 1] = 2 - va_ratios_up_c[va_ratios_up_c < 1]
      va_ratios_down_c[va_ratios_down_c > 1] = 2 - va_ratios_down_c[va_ratios_down_c > 1]

      vb_ratios_up_c = vb_ratios.clone()
      vb_ratios_down_c = vb_ratios.clone()
      vb_ratios_up_c[vb_ratios_up_c < 1] = 2 - vb_ratios_up_c[vb_ratios_up_c < 1]
      vb_ratios_down_c[vb_ratios_down_c > 1] = 2 - vb_ratios_down_c[vb_ratios_down_c > 1]


      pa_ratios_up_c = self.pa_ratio_up_layer(pa_ratios_up_c - 1)
      pa_ratios_down_c = self.pa_ratio_down_layer(pa_ratios_down_c - 1)

      pb_ratios_up_c = self.pb_ratio_up_layer(pb_ratios_up_c - 1)
      pb_ratios_down_c = self.pb_ratio_down_layer(pb_ratios_down_c - 1)

      va_ratios_up_c = self.va_ratio_up_layer(va_ratios_up_c - 1)
      va_ratios_down_c = self.va_ratio_down_layer(va_ratios_down_c - 1)

      vb_ratios_up_c = self.vb_ratio_up_layer(vb_ratios_up_c - 1)
      vb_ratios_down_c = self.vb_ratio_down_layer(vb_ratios_down_c - 1)

      # dim: b x 1 x L x T
      # b x t x s x l * b x t x s x l
      pa_ratios_up = pa_ratios_up_c.repeat((1, self.S, 1, 1)).transpose(1, 3).transpose(2, 3) + self.pa_transition_weights[:, :, :, 0, 0].unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
      pa_ratios_down = pa_ratios_down_c.repeat((1, self.S, 1, 1)).transpose(1, 3).transpose(2, 3) + self.pa_transition_weights[:, :, :, 0, 1].unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
      pa_ratios = torch.cat((pa_ratios_up.unsqueeze(4), pa_ratios_down.unsqueeze(4)), dim=4).unsqueeze(4)
      
      pb_ratios_up = pb_ratios_up_c.repeat((1, self.S, 1, 1)).transpose(1, 3).transpose(2, 3) + self.pb_transition_weights[:, :, :, 0, 0].unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
      pb_ratios_down = pb_ratios_down_c.repeat((1, self.S, 1, 1)).transpose(1, 3).transpose(2, 3) + self.pb_transition_weights[:, :, :, 0, 1].unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
      pb_ratios = torch.cat((pb_ratios_up.unsqueeze(4), pb_ratios_down.unsqueeze(4)), dim=4).unsqueeze(4)
      
      va_ratios_up = va_ratios_up_c.repeat((1, self.S, 1, 1)).transpose(1, 3).transpose(2, 3) + self.va_transition_weights[:, :, :, 0, 0].unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
      va_ratios_down = va_ratios_down_c.repeat((1, self.S, 1, 1)).transpose(1, 3).transpose(2, 3) + self.va_transition_weights[:, :, :, 0, 1].unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
      va_ratios = torch.cat((va_ratios_up.unsqueeze(4), va_ratios_down.unsqueeze(4)), dim=4).unsqueeze(4)
      
      vb_ratios_up = vb_ratios_up_c.repeat((1, self.S, 1, 1)).transpose(1, 3).transpose(2, 3) + self.vb_transition_weights[:, :, :, 0, 0].unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
      vb_ratios_down = vb_ratios_down_c.repeat((1, self.S, 1, 1)).transpose(1, 3).transpose(2, 3) + self.vb_transition_weights[:, :, :, 0, 1].unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
      vb_ratios = torch.cat((vb_ratios_up.unsqueeze(4), vb_ratios_down.unsqueeze(4)), dim=4).unsqueeze(4)

      # Probabilities
      pa_probs = self.get_transition_matrix(pa).unsqueeze(1).repeat((1, self.S, 1, 1, 1)) @ torch.softmax(self.pa_transition_probs, dim=3).unsqueeze(0).repeat((x.shape[0], 1, 1, 1, 1))
      pb_probs = self.get_transition_matrix(pb).unsqueeze(1).repeat((1, self.S, 1, 1, 1)) @ torch.softmax(self.pb_transition_probs, dim=3).unsqueeze(0).repeat((x.shape[0], 1, 1, 1, 1))
      va_probs = self.get_transition_matrix(va).unsqueeze(1).repeat((1, self.S, 1, 1, 1)) @ torch.softmax(self.va_transition_probs, dim=3).unsqueeze(0).repeat((x.shape[0], 1, 1, 1, 1))
      vb_probs = self.get_transition_matrix(vb).unsqueeze(1).repeat((1, self.S, 1, 1, 1)) @ torch.softmax(self.vb_transition_probs, dim=3).unsqueeze(0).repeat((x.shape[0], 1, 1, 1, 1))

      return ([pa_init  , pb_init  , va_init  , vb_init], 
              [pa_ratios, pb_ratios, va_ratios, vb_ratios], 
              [pa_probs , pb_probs , va_probs , vb_probs])
 
 
    def forward(self, x, spreads=False, debug=False, fts_only=False, s_values=False):
        out = self.main(x)
 
        if fts_only:
          return out
 
        S_values = self.S_values(out)
        
        if not self.prediction_mode:
          init_values = self.init_values(out)

          # Should have dim L
          p_a = init_values[:, :, :, 0]
          p_b = init_values[:, :, :, 1]
          v_a = init_values[:, :, :, 2]
          v_b = init_values[:, :, :, 3]
 
        # Should have dim S
        p_a_inc_weights = S_values[:, :, :, 0]
        p_b_inc_weights = S_values[:, :, :, 3]
        v_a_inc_weights = S_values[:, :, :, 6]
        v_b_inc_weights = S_values[:, :, :, 9]
 
        p_a_dec_weights = S_values[:, :, :, 1]
        p_b_dec_weights = S_values[:, :, :, 4]
        v_a_dec_weights = S_values[:, :, :, 7]
        v_b_dec_weights = S_values[:, :, :, 10]
 
        p_a_prob_weights = S_values[:, :, :, 2]
        p_b_prob_weights = S_values[:, :, :, 5]
        v_a_prob_weights = S_values[:, :, :, 8]
        v_b_prob_weights = S_values[:, :, :, 11]

        pa_transition_weights = self.pa_transition_weights
        pb_transition_weights = self.pb_transition_weights
        va_transition_weights = self.va_transition_weights
        vb_transition_weights = self.vb_transition_weights

        pa_transition_probs = self.pa_transition_probs
        pb_transition_probs = self.pb_transition_probs
        va_transition_probs = self.va_transition_probs
        vb_transition_probs = self.vb_transition_probs

        if self.prediction_mode:
          inits, ratios, probs = self.extract_values(x)
          p_a, p_b, v_a, v_b = inits
          pa_transition_weights, pb_transition_weights, va_transition_weights, vb_transition_weights = ratios
          pa_transition_probs, pb_transition_probs, va_transition_probs, vb_transition_probs = probs
        
        # Every matrix is 10 x 50
        pa_order_preds, pa_wr = self.update_order_matrix(p_a, pa_transition_weights, pa_transition_probs, p_a_inc_weights, p_a_dec_weights, p_a_prob_weights, debug)
        pb_order_preds, pb_wr = self.update_order_matrix(p_b, pb_transition_weights, pb_transition_probs, p_b_inc_weights, p_b_dec_weights, p_b_prob_weights, debug)
        va_order_preds, va_wr = self.update_order_matrix(v_a, va_transition_weights, va_transition_probs, v_a_inc_weights, v_a_dec_weights, v_a_prob_weights, debug)
        vb_order_preds, vb_wr = self.update_order_matrix(v_b, vb_transition_weights, vb_transition_probs, v_b_inc_weights, v_b_dec_weights, v_b_prob_weights, debug)
 
        w_ratios = [pa_wr, pb_wr, va_wr, vb_wr]
 
        orderbook = self.construct_order_book_matrix(pa_order_preds, pb_order_preds, va_order_preds, vb_order_preds)
 
        if spreads:
            bid_ask_spreads = pa_order_preds - pb_order_preds
            ask_level_spreads = pa_order_preds[:, :, self.ranget_next_level_all, self.ranget_col_all] - pa_order_preds[:, :, self.ranget_level_all, self.ranget_col_all]
            bid_level_spreads = pb_order_preds[:, :, self.ranget_level_all, self.ranget_col_all] - pb_order_preds[:, :, self.ranget_next_level_all, self.ranget_col_all]
            weight_matrices = [
                               self.pa_transition_weights[:, :, :, :, 0].squeeze(3).transpose(0,2).transpose(0,1), 
                               self.pa_transition_weights[:, :, :, :, 1].squeeze(3).transpose(0,2).transpose(0,1), 
                               self.pa_transition_probs.reshape((self.S, self.L, 9)),
 
                               self.pb_transition_weights[:, :, :, :, 0].squeeze(3).transpose(0,2).transpose(0,1), 
                               self.pb_transition_weights[:, :, :, :, 1].squeeze(3).transpose(0,2).transpose(0,1), 
                               self.pb_transition_probs.reshape((self.S, self.L, 9)),
 
                               self.va_transition_weights[:, :, :, :, 0].squeeze(3).transpose(0,2).transpose(0,1),
                               self.va_transition_weights[:, :, :, :, 1].squeeze(3).transpose(0,2).transpose(0,1),
                               self.va_transition_probs.reshape((self.S, self.L, 9)),
 
                               self.vb_transition_weights[:, :, :, :, 0].squeeze(3).transpose(0,2).transpose(0,1), 
                               self.vb_transition_weights[:, :, :, :, 1].squeeze(3).transpose(0,2).transpose(0,1), 
                               self.vb_transition_probs.reshape((self.S, self.L, 9))
                               ]
            if debug:
              return orderbook, bid_ask_spreads, ask_level_spreads, bid_level_spreads, weight_matrices, w_ratios, out
 
            return orderbook, bid_ask_spreads, ask_level_spreads, bid_level_spreads, weight_matrices
            
        if debug:
            return orderbook, w_ratios, out

        if s_values:
          return orderbook, [p_a_inc_weights, p_b_inc_weights, v_a_inc_weights, v_b_inc_weights], [p_a_dec_weights, p_b_dec_weights, v_a_dec_weights, v_b_dec_weights], [p_a_prob_weights, p_b_prob_weights, v_a_prob_weights, v_b_prob_weights]
        
        return orderbook