from models.generator import Generator
import torchvision.transforms as transforms
import torch.utils.data as data
from helper.dataset import OrderBookDataSet
import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_seed = 54231856
torch.manual_seed(exp_seed)
np.random.seed(exp_seed)

# -------------------------------------------------
# EVALUATION: PREDICTION
# -------------------------------------------------
T = 35
P = 15

# The generator should be loaded in from an existing model
S = 16
sampling_rate = 3
gamma = 0.99
batch_size = 1024
weight_div_lambda = 0.001
gen = Generator(1, 100, 64, S=S, T=15, L=10, gamma=gamma, sampling_rate=sampling_rate, prediction_mode=True)
noise_dim = 100
gdrive_path = '/content/drive/MyDrive/MScThesis/'

# FI-2010
test_dataset_size = 4
test_data_path = gdrive_path + "Data/FI-2010/3.NoAuction_DecPre/NoAuction_DecPre_Testing/Test_Dst_NoAuction_DecPre_CF_" + str(test_dataset_size) + ".txt"

# BTCUSDT
# test_dataset_size = -1
# test_data_path = data_path = gdrive_path + "Data/Crypto/BTCUSDT_01_03_2021_1926_14_03_2021_0107.txt"

test_my_dataset = OrderBookDataSet(test_data_path, transforms.Compose([ transforms.ToTensor() ]), T=T, size=test_dataset_size, prediction_horizon=P, set_type="test")
test_loader = data.DataLoader(test_my_dataset , batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)

mse_loss = nn.MSELoss()
total_loss = 0
price_loss = 0
volume_loss = 0

price_indices = torch.arange(0, P) * 2
volume_indices = torch.arange(0, P) * 2 + 1

inc_all = []
dec_all = []
prob_all = []

gen.eval()
with torch.no_grad():
  for idx, real_samples in enumerate(test_loader):
    base_rs = real_samples[0].to(device)
    pred_rs = real_samples[1].to(device)
    pred_fake, incw, decw, probw = gen(base_rs, s_values=True)
    total_loss += mse_loss(pred_fake, pred_rs)
    price_loss += mse_loss(pred_fake[:, :, :, price_indices], pred_rs[:, :, :, price_indices])
    volume_loss += mse_loss(pred_fake[:, :, :, volume_indices], pred_rs[:, :, :, volume_indices])

    inc_all.append(incw[0])
    dec_all.append(decw[0])
    prob_all.append(probw[0])

print("Total loss:", str(total_loss.item()))
print("Mean loss:", str(total_loss.item() / len(test_loader)))
print("Price loss:", str(price_loss.item() / len(test_loader)))
print("Volume loss:", str(volume_loss.item() / len(test_loader)))

p_a_inc = torch.softmax(torch.cat(inc_all), dim=2).mean(dim=0)[0]
p_a_dec = torch.softmax(torch.cat(dec_all), dim=2).mean(dim=0)[0]
p_a_prob = torch.softmax(torch.cat(prob_all), dim=2).mean(dim=0)[0]
