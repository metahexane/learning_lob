# -------------------------------------------------
# EVALUATION: GENERATION
# -------------------------------------------------
import torch
import numpy as np
import torchvision.transforms as transforms
from helper.dataset import OrderBookDataSet
from models.generator import Generator
from scipy.stats import ks_2samp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_seed = 54231856
torch.manual_seed(exp_seed)
np.random.seed(exp_seed)

# The generator should be loaded in from an existing model
S = 16
L = 10
T = 50
sampling_rate = 3
gamma = 0.925
weight_div_lambda = 0.001
gen = Generator(1, 100, 64, S=S, T=T, L=L, gamma=gamma, sampling_rate=sampling_rate, prediction_mode=True)
noise_dim = 100

gdrive_path = '/content/drive/MyDrive/MScThesis/'
data_path = gdrive_path + "Data/Crypto/BTCUSDT_01_03_2021_1926_14_03_2021_0107.txt"
dataset_size = -1
my_dataset = OrderBookDataSet(data_path, transforms.Compose([ transforms.ToTensor() ]), T=T, size=dataset_size)

# -------------------------------------------------
# Calculation of the Axiom losses
# -------------------------------------------------
true_data = my_dataset.get_data().reshape((-1, 40))
mus = [16384]
VOL_EPS = 1e-4

gen.eval()
with torch.no_grad():
  for mu in mus:
    fixed_noise = torch.randn(mu, noise_dim, 1, 1, device=device)
    # Input `fake'
    o = gen(fixed_noise)
    img = o.detach().cpu()

    # --------------------------
    # Indices
    # --------------------------

    level_range = torch.arange(0, L)
    all_levels = level_range.repeat(int(T/2))

    # Axiom 1:
    price_ask_range = (level_range * 2).repeat(T)
    price_bid_range = (level_range * 2 + 1).repeat(T)

    # Axiom 2:
    level_sub_pa_range = (torch.arange(0, int(L/2)) * 4).repeat(T) # 0, 4, 8, 12, 16, ...
    level_post_pa_range = (torch.arange(0, int(L/2)) * 4 + 2).repeat(T) # 2, 6, 10, 14, ....

    level_sub_pb_range = (torch.arange(0, int(L/2)) * 4 + 1).repeat(T) # 1, 5, 9, 13, 17, ...
    level_post_pb_range = (torch.arange(0, int(L/2)) * 4 + 3).repeat(T) # 3, 7, 11, 15, ...

    time_range = torch.arange(0, T)
    price_range = (time_range * 2).repeat_interleave(L)
    volume_range = (time_range * 2 + 1).repeat_interleave(L)
    price_level_range = (time_range * 2).repeat_interleave(int(L/2))

    # Axiom 3
    price_sub_range = (torch.arange(0, int(T/2)) * 4).repeat_interleave(L) # 0, 4, 8, 12, 16 ....
    price_post_range = (torch.arange(0, int(T/2)) * 4 + 2).repeat_interleave(L) # 2, 6, 10, 14, ....

    # --------------------------
    # Calculations for Distros
    # --------------------------
    ask_prices = img[:, 0, price_ask_range, price_range].reshape((mu, L, T))
    bid_prices = img[:, 0, price_bid_range, price_range].reshape((mu, L, T))
    ask_prices = (ask_prices - ask_prices.mean(2, keepdim=True))# / ask_prices.std(2, keepdim=True)
    bid_prices = (bid_prices - bid_prices.mean(2, keepdim=True))# / bid_prices.std(2, keepdim=True)

    # Return Dist, no need for location normalization
    time_sub_range = torch.arange(0, T - 1)
    time_post_range = torch.arange(1, T)
    level_range_ = torch.arange(0, L)

    mid_prices = (ask_prices + bid_prices) / 2

    old_prices = mid_prices[:, :, time_sub_range]
    new_prices = mid_prices[:, :, time_post_range]
    returns = (new_prices - old_prices) / old_prices
    top_level_returns = returns[:, 0, :] # Use for calculation

    true_ask = true_data[:, 0]
    true_bid = true_data[:, 2]
    true_midprices = (true_ask + true_bid) / 2
    t_sub_price_range = torch.arange(0, true_midprices.shape[0] - 1)
    t_post_price_range = torch.arange(1, true_midprices.shape[0])
    true_returns = (true_midprices[t_post_price_range] - true_midprices[t_sub_price_range]) / true_midprices[t_sub_price_range]
    D_returns, _ = ks_2samp(top_level_returns.flatten(), true_returns)


    # Price Dist
    # Calculate difference in distribution for every level, then average it out
    ask_distances = []
    bid_distances = []
    for i in range(L):
      gen_ask_prices = ask_prices[:, i, :].flatten()
      gen_bid_prices = bid_prices[:, i, :].flatten()

      true_ask_price_range = i * 4 # 0, 4, 8, 
      true_bid_price_range = i * 4 + 2 # 2, 6, 10, 

      true_ask_prices = true_data[:, true_ask_price_range]
      true_bid_prices = true_data[:, true_bid_price_range]

      true_ask_prices = (true_ask_prices - true_ask_prices.mean())# / true_ask_prices.std()
      true_bid_prices = (true_bid_prices - true_bid_prices.mean())# / true_bid_prices.std()

      D_ask, _ = ks_2samp(gen_ask_prices, true_ask_prices)
      D_bid, _ = ks_2samp(gen_bid_prices, true_bid_prices)

      ask_distances.append(D_ask)
      bid_distances.append(D_bid)
    
    ask_distance = torch.Tensor(ask_distances).mean()
    bid_distance = torch.Tensor(bid_distances).mean()


    # Volume Dist - no need for Z/location normalization
    ask_volumes = img[:, 0, price_ask_range, volume_range].reshape((mu, L, T))
    bid_volumes = img[:, 0, price_bid_range, volume_range].reshape((mu, L, T))

    vol_ask_errors = []
    vol_bid_errors = []

    for i in range(L):
      # Gens
      gen_ask_volumes = ask_volumes[:, i, :]
      gen_bid_volumes = bid_volumes[:, i, :]

      volume_sub_range = torch.arange(0, gen_ask_volumes.shape[1] - 1)
      volume_post_range = torch.arange(1, gen_bid_volumes.shape[1])

      gen_vol_ask_up = gen_ask_volumes[:, volume_post_range] > gen_ask_volumes[:, volume_sub_range]
      gen_vol_ask_stag = torch.abs(gen_ask_volumes[:, volume_post_range] - gen_ask_volumes[:, volume_sub_range]) < VOL_EPS
      gen_vol_ask_down = gen_ask_volumes[:, volume_post_range] < gen_ask_volumes[:, volume_sub_range]

      gen_vol_bid_up = gen_bid_volumes[:, volume_post_range] > gen_bid_volumes[:, volume_sub_range]
      gen_vol_bid_stag = torch.abs(gen_bid_volumes[:, volume_post_range] - gen_bid_volumes[:, volume_sub_range]) < VOL_EPS
      gen_vol_bid_down = gen_bid_volumes[:, volume_post_range] < gen_bid_volumes[:, volume_sub_range]

      gen_vol_ask_up = gen_vol_ask_up.sum(1) / gen_vol_ask_up.shape[1]
      gen_vol_ask_down = gen_vol_ask_down.sum(1) / gen_vol_ask_down.shape[1]
      gen_vol_ask_stag = gen_vol_ask_stag.sum(1) / gen_vol_ask_stag.shape[1]

      gen_vol_ask_up = gen_vol_ask_up.mean()
      gen_vol_ask_down = gen_vol_ask_down.mean()
      gen_vol_ask_stag = gen_vol_ask_stag.mean()

      gen_vol_bid_up = gen_vol_bid_up.sum(1) / gen_vol_bid_up.shape[1]
      gen_vol_bid_stag = gen_vol_bid_stag.sum(1) / gen_vol_bid_stag.shape[1]
      gen_vol_bid_down = gen_vol_bid_down.sum(1) / gen_vol_bid_down.shape[1]

      gen_vol_bid_up = gen_vol_bid_up.mean()
      gen_vol_bid_stag = gen_vol_bid_stag.mean()
      gen_vol_bid_down = gen_vol_bid_down.mean()

      # True
      true_ask_price_range = i * 4 + 1 # 1, 5, 9, 
      true_bid_price_range = i * 4 + 3 # 3, 7, 11, 

      true_ask_volumes = true_data[:, true_ask_price_range]
      true_bid_volumes = true_data[:, true_bid_price_range]

      volume_sub_range = torch.arange(0, true_ask_volumes.shape[0] - 1)
      volume_post_range = torch.arange(1, true_ask_volumes.shape[0])

      true_vol_ask_up = true_ask_volumes[volume_post_range] > true_ask_volumes[volume_sub_range]
      true_vol_ask_stag = true_ask_volumes[volume_post_range] == true_ask_volumes[volume_sub_range]
      true_vol_ask_down = true_ask_volumes[volume_post_range] < true_ask_volumes[volume_sub_range]

      true_vol_bid_up = true_bid_volumes[volume_post_range] > true_bid_volumes[volume_sub_range]
      true_vol_bid_stag = true_bid_volumes[volume_post_range] == true_bid_volumes[volume_sub_range]
      true_vol_bid_down = true_bid_volumes[volume_post_range] < true_bid_volumes[volume_sub_range]

      true_vol_ask_up = true_vol_ask_up.sum() / true_vol_ask_up.shape[0]
      true_vol_ask_stag = true_vol_ask_stag.sum() / true_vol_ask_stag.shape[0]
      true_vol_ask_down = true_vol_ask_down.sum() / true_vol_ask_down.shape[0]

      true_vol_bid_up = true_vol_bid_up.sum() / true_vol_bid_up.shape[0]
      true_vol_bid_stag = true_vol_bid_stag.sum() / true_vol_bid_stag.shape[0]
      true_vol_bid_down = true_vol_bid_down.sum() / true_vol_bid_down.shape[0]

      vol_ask_errors.append((gen_vol_ask_up - true_vol_ask_up)**2)
      vol_ask_errors.append((gen_vol_ask_stag - true_vol_ask_stag)**2)
      vol_ask_errors.append((gen_vol_ask_down - true_vol_ask_down)**2)

      vol_bid_errors.append((gen_vol_bid_up - true_vol_bid_up)**2)
      vol_bid_errors.append((gen_vol_bid_stag - true_vol_bid_stag)**2)
      vol_bid_errors.append((gen_vol_bid_down - true_vol_bid_down)**2)

    vol_ask_errors = torch.Tensor(vol_ask_errors).mean()
    vol_bid_errors = torch.Tensor(vol_bid_errors).mean()

    # --------------------------
    # Calculations for Axioms
    # --------------------------

    # Axiom 1
    valid_axiom_1 = img[:, 0, price_ask_range, price_range] - img[:, 0, price_bid_range, price_range] > 0
    valid_axiom_1_score = torch.sum(valid_axiom_1, 1) / valid_axiom_1.shape[1]
    valid_axiom_1_score_mean = torch.mean(valid_axiom_1_score)

    # Axiom 2
    valid_axiom_2_pa = img[:, 0, level_post_pa_range, price_level_range] - img[:, 0, level_sub_pa_range, price_level_range] > 0
    valid_axiom_2_pa_score = torch.sum(valid_axiom_2_pa, 1) / valid_axiom_2_pa.shape[1]
    valid_axiom_2_pa_score_mean = torch.mean(valid_axiom_2_pa_score)

    valid_axiom_2_pb = img[:, 0, level_sub_pb_range, price_level_range] - img[:, 0, level_post_pb_range, price_level_range] > 0
    valid_axiom_2_pb_score = torch.sum(valid_axiom_2_pb, 1) / valid_axiom_2_pa.shape[1]
    valid_axiom_2_pb_score_mean = torch.mean(valid_axiom_2_pb_score)

    # Axiom 3
    avg_change = ((img[:, 0, all_levels, price_sub_range] / img[:, 0, all_levels, price_post_range]) - 1).mean(1)
    axiom_3_mean = avg_change.mean()

    total_loss = (valid_axiom_1_score_mean + valid_axiom_2_pa_score_mean + valid_axiom_2_pb_score_mean) / 3

    a1_item = str(valid_axiom_1_score_mean.item())
    a2_pa_item = str(valid_axiom_2_pa_score_mean.item())
    a2_pb_item = str(valid_axiom_2_pb_score_mean.item())
    a12_item = str(total_loss.item())
    a3_item = str(axiom_3_mean.item())

    returns_item = str(D_returns)
    ask_distance_item = str(ask_distance.item())
    bid_distance_item = str(bid_distance.item())
    vol_ask_error_item = str(vol_ask_errors.item())
    vol_bid_error_item = str(vol_bid_errors.item())

    print(a1_item + "	" + a2_pa_item + "	" + a2_pb_item + "	" + a12_item + "	" + a3_item + "	" + returns_item + "	" + ask_distance_item + "	" + bid_distance_item + "	" + vol_ask_error_item + "	" + vol_bid_error_item)