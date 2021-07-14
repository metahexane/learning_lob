import torch
from sklearn.preprocessing import minmax_scale
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm_notebook
import torch.utils.data as data
import torchvision.transforms as transforms
from helper.dataset import OrderBookDataSet
from models.cnn import CNNPred
from models.mlp import MLP
from models.generator import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_seed = 54231856
torch.manual_seed(exp_seed)
np.random.seed(exp_seed)

def transform_data(data, stock_indices):
  s1 = data[:stock_indices[0]]
  s2 = data[stock_indices[0]:stock_indices[1], :]
  s3 = data[stock_indices[1]:stock_indices[2], :]
  s4 = data[stock_indices[2]:stock_indices[3], :]
  s5 = data[stock_indices[3]:, :]

  s1_valid = minmax_scale(s1.reshape((-1, 2)), axis=0).reshape((-1, 40, 1))
  s2_valid = minmax_scale(s2.reshape((-1, 2)), axis=0).reshape((-1, 40, 1))
  s3_valid = minmax_scale(s3.reshape((-1, 2)), axis=0).reshape((-1, 40, 1))
  s4_valid = minmax_scale(s4.reshape((-1, 2)), axis=0).reshape((-1, 40, 1))
  s5_valid = minmax_scale(s5.reshape((-1, 2)), axis=0).reshape((-1, 40, 1))

  return np.concatenate((s1_valid, s2_valid, s3_valid, s4_valid, s5_valid), axis=0)

dataset_type = "btcusdt"
gdrive_path = '/content/drive/MyDrive/MScThesis/'
fi2010_size = 4
vpc = 20 # 40 for 1d

if dataset_type == "fi2010":
  # FI-2010
  # Train dataset
  dataset_size = fi2010_size
  data_path = gdrive_path + "Data/FI-2010/3.NoAuction_DecPre/NoAuction_DecPre_Training/Train_Dst_NoAuction_DecPre_CF_" + str(dataset_size) + ".txt"
              
  # Test dataset
  test_dataset_size = fi2010_size
  test_data_path = gdrive_path + "Data/FI-2010/3.NoAuction_DecPre/NoAuction_DecPre_Testing/Test_Dst_NoAuction_DecPre_CF_" + str(test_dataset_size) + ".txt"

elif dataset_type == "btcusdt":
  # BTCUSDT
  # Train dataset
  data_path = gdrive_path + "Data/Crypto/BTCUSDT_01_03_2021_1926_14_03_2021_0107.txt"
  dataset_size = -1
  # BTCUSDT
  test_dataset_size = -1
  test_data_path = data_path = gdrive_path + "Data/Crypto/BTCUSDT_01_03_2021_1926_14_03_2021_0107.txt"

my_dataset = OrderBookDataSet(data_path, transforms.Compose([ transforms.ToTensor() ]), vpc=vpc, T=35, size=dataset_size, prediction_horizon=15)
train_loader = data.DataLoader(my_dataset , batch_size=1024, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)

test_my_dataset = OrderBookDataSet(test_data_path, transforms.Compose([ transforms.ToTensor() ]), vpc=vpc, T=35, size=test_dataset_size, prediction_horizon=15, set_type="test")
test_loader = data.DataLoader(test_my_dataset , batch_size=1024, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)

network = "obcnn" # [obcnn/mlp/cnn]
load = False
model_name = network.upper() + "-" + dataset_type.upper() + "-35E-PRED-FINAL" # [WGAN/DCGAN/OBGAN]-[FI2010/BTCUSDT]-[#EPOCHS]E-[GEN/PRED]-FINAL

if network == "mlp":
  # MLP
  mlp = MLP()
  input_dim = 1
elif network == "cnn":
  # CNN
  mlp = CNNPred(64)
  input_dim = 2
elif network == "obcnn":
  # OBCNN
  input_dim = 2
  S = 16 # 16 # est. distros.
  sampling_rate = 3
  gamma = 0.925
  weight_div_lambda = 0.001
  mlp = Generator(1, 100, 64, S=S, T=15, L=10, gamma=gamma, sampling_rate=sampling_rate, prediction_mode=True)

  ba_criterion = nn.BCELoss()
  al_criterion = nn.BCELoss()
  bl_criterion = nn.BCELoss()
  weight_div_criterion = nn.KLDivLoss(reduction='batchmean')

  L = 10
  batch_size = 1024
  al_bl_size = int((L - 1) * 15)
  target_ba = torch.ones((batch_size, 1, L, 15), device=device)
  target_al_bl = torch.ones((batch_size, 1, al_bl_size), device=device)
mlp.to(device)

adam = optim.Adam(mlp.parameters(), weight_decay=0.001)
crit = nn.MSELoss()
all_losses = []
its = 0

if load:
  loaded = torch.load(gdrive_path + "Models/" + model_name + ".ptx")
  mlp.load_state_dict(loaded['model'])
  adam.load_state_dict(loaded['optim'])
  all_losses = loaded['losses']
  its = loaded['iters']

epochs = 35
eps = 1e-10
torch.autograd.set_detect_anomaly(True)
def ext_sigmoid(x, k):
  return 1 / (1 + torch.exp(- k * x))
mlp.train()

for epoch in tqdm_notebook(range(epochs)):
  for idx, real_samples in enumerate(train_loader): 
    adam.zero_grad()

    input = real_samples[0].to(device)
    real = real_samples[1].to(device)
    if input_dim == 1:
      input = input.squeeze(1).reshape((real_samples[0].shape[0], -1))
      real = real.squeeze(1).reshape((real_samples[1].shape[0], -1))

    if network == "obcnn":
      pred, _, _, _, _, _, _ = mlp(input, spreads=True, debug=True)
      loss = crit(pred, real)

    else:
      pred = mlp(input)
      loss = crit(pred, real)

    loss.backward()
    adam.step()

    all_losses.append(loss.item())

    if its % 200 == 0:
      print("[" + str(its) + "] Loss:", loss.item())

    its += 1
  torch.save({
        'model': mlp.state_dict(),
        'optim': adam.state_dict(),
        'losses': all_losses,
        'iters': its
    }, gdrive_path + "Models/" + model_name + ".ptx")

