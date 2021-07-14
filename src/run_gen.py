import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm_notebook
import torch.utils.data as data
import torchvision.transforms as transforms
from helper.dataset import OrderBookDataSet
from models.generator import Generator
from models.default_generator import DefaultGenerator
from models.discriminator import Discriminator
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_seed = 54231856
torch.manual_seed(exp_seed)
np.random.seed(exp_seed)

# ----------------------------------------
# GENERATIVE ADVERSARIAL NETWORK
# Model Format:
# [WGAN/DCGAN/OBGAN]-[FI2010/BTCUSDT]-[#EPOCHS]E-[GEN/PRED]-FINAL
# ----------------------------------------
 
load_model = True
model_suffix = "DCGAN-BTCUSDT-50E-GEN-FINAL" #-" + str(exp_seed) # 3-Mod3, 3-Mod4, 4-Mod1, 4-Mod2, 4-Mod3-GP

model_params = model_suffix.lower().split("-")
use_obgan = model_params[0] == "obgan" # Set to True if Generator is used instead of DefaultGenerator
use_dcgan_loss = model_params[0] == "dcgan"

dataset_type = model_params[1] # fi2010/btcusdt
fi2010_size = 4

batch_size = 1024

if model_params[3] == "gen":
  # Generation params
  L = 10
  T = 50
  P = 0
else:
  # Prediction params
  L = 10 # levels
  T = 35 # time frame
  P = 15 # prediction horizon

S = 16 # 16 # est. distros.
sampling_rate = 3
gamma = 0.925

in_dim = 1
fm_g = 64
fm_d = 64
noise_dim = 100

# Params for Adam
beta1 = 0
beta2 = 0.9
gp_lambda = 10
lr = 0.001

l2_reg_lambda = 0.001
weight_div_lambda = 0.001

# ----------------------------------------------
# TRAIN DATA SET
# ----------------------------------------------

# TO CHANGE DATA CHANGE PATH AND SET SIZE TO CORRESPONDING SIZE
gdrive_path = '/content/drive/MyDrive/MScThesis/'

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
  test_data_path = gdrive_path + "Data/Crypto/BTCUSDT_01_03_2021_1926_14_03_2021_0107.txt"



prediction_mode = P > 0
my_dataset = OrderBookDataSet(data_path, transforms.Compose([ transforms.ToTensor() ]), T=T, size=dataset_size, prediction_horizon=P)
train_loader = data.DataLoader(my_dataset , batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)

# ----------------------------------------------
# TEST DATA SET
# ----------------------------------------------

test_my_dataset = OrderBookDataSet(test_data_path, transforms.Compose([ transforms.ToTensor() ]), T=T, size=test_dataset_size, prediction_horizon=P, set_type="test")
test_loader = data.DataLoader(test_my_dataset , batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)
mse_loss = nn.MSELoss()

# ----------------------------------------------
# INITIALIZE MODELS
# ----------------------------------------------
gen_T = P if prediction_mode else T
if use_obgan:
  gen = Generator(in_dim, noise_dim, fm_g, S=S, T=gen_T, L=L, gamma=gamma, sampling_rate=sampling_rate, prediction_mode=prediction_mode).to(device) # Generator/DefaultGenerator
else:
  gen = DefaultGenerator(in_dim, noise_dim, fm_g, prediction_mode=prediction_mode).to(device) # Generator/DefaultGenerator

dis = Discriminator(in_dim, fm_d, out_sigmoid=use_dcgan_loss).to(device)

if load_model:
    loaded = torch.load(gdrive_path + "Models/" + model_suffix + ".ptx")
    gen.load_state_dict(loaded['generator'])
    dis.load_state_dict(loaded['discriminator'])
 
dc_loss = nn.BCELoss()
# Initialize BCELoss function
ba_criterion = nn.BCELoss()
al_criterion = nn.BCELoss()
bl_criterion = nn.BCELoss()
weight_div_criterion = nn.KLDivLoss(reduction='batchmean')
 
# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
 
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(dis.parameters(), lr=lr, betas=(beta1,beta2))
optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=(beta1,beta2))
 
if load_model:
    optimizerD.load_state_dict(loaded['opt_dis'])
    optimizerG.load_state_dict(loaded['opt_gen'])
 
# Lists to keep track of progress
G_losses = []
D_losses = []
 
G_gan_loss = []
G_ba_loss = []
G_al_loss = []
G_bl_loss = []
G_wdiv_loss = []
D_fts_losses = []

test_mean_mse_losses = []
test_price_mse_losses = []
test_volume_mse_losses = []

iters = 0
 
if load_model:
  G_losses = loaded['loss_generator']
  D_losses = loaded['loss_discriminator'] 
  G_gan_loss = loaded['loss_gan']
  G_ba_loss = loaded['loss_bid_ask']
  G_al_loss = loaded['loss_ask_level']
  G_bl_loss = loaded['loss_bid_level']
  G_wdiv_loss = loaded['loss_wdiv']
  iters = loaded['iters']
  test_mean_mse_losses = loaded['mean_mse_losses']
  test_price_mse_losses = loaded['price_mse_losses']
  test_volume_mse_losses = loaded['volume_mse_losses']
 
al_bl_size = int((L - 1) * gen_T)
target_ba = torch.ones((batch_size, 1, L, gen_T), device=device)
target_al_bl = torch.ones((batch_size, 1, al_bl_size), device=device)

num_epochs = 50
fixed_noise = torch.randn(1, noise_dim, 1, 1, device=device)
n_critic = 5

# Wasserstein GAN
eps = 1e-10
torch.autograd.set_detect_anomaly(True)
def ext_sigmoid(x, k):
  return 1 / (1 + torch.exp(- k * x))  

gen.train()
dis.train()
 
print("Starting Training Loop...")
for j in tqdm_notebook(range(num_epochs)):
    gen.train()
    dis.train()
    for idx, real_samples in enumerate(train_loader): 
        ############################
        # (1) Update D network
        ############################
        if prediction_mode:
          base_rs = real_samples[0].to(device)
          pred_rs = real_samples[1].to(device)
          rs = Variable(torch.cat((base_rs, pred_rs), dim=3))
        else:
          rs = Variable(real_samples.to(device))
 
        dis.zero_grad()
        output_real = dis(rs).view(-1)
 
        if use_dcgan_loss:
          label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
          real_loss = dc_loss(output_real, label)
          real_loss.backward()
 
        if prediction_mode:
          noise = Variable(base_rs)
          gen_out = gen(noise, debug=True) # B x 1 x 20 x P
        else:
          noise = Variable(torch.randn(batch_size, noise_dim, 1, 1, device=device))
          gen_out = gen(noise, debug=True)
        
        if use_obgan:
          fake = gen_out[0].detach()
        else:
          fake = gen_out.detach()

        if prediction_mode:
          fake = torch.cat((noise, fake), dim=3)
        
        output_fake = dis(fake).view(-1)
 
        if not use_dcgan_loss:
          gp_noise = torch.rand(batch_size, 1, 1, 1, device=device)
          gp_sample = gp_noise * rs + (1 - gp_noise) * fake
          gp_sample = Variable(gp_sample, requires_grad=True).to(device)
          gp_probs = dis(gp_sample)
          gp_gradient = torch_grad(gp_probs, gp_sample, torch.ones(gp_probs.size(), device=device), True, True)[0]
          gp_gradient = gp_gradient.view(batch_size, -1)
          gradients_norm = torch.sqrt(torch.sum(gp_gradient ** 2, dim=1) + 1e-12)
          wdistance = output_fake.mean() - output_real.mean()
  
          errD = wdistance + gp_lambda * ((gradients_norm - 1) ** 2).mean()
          errD.backward()
        else:
          label.fill_(fake_label)
          fake_loss = dc_loss(output_fake, label)
          fake_loss.backward()
          errD = real_loss + fake_loss
        
        optimizerD.step()
 
        if (not use_dcgan_loss and iters % n_critic == 0) or use_dcgan_loss:
 
            ###########################
            # (2) Update G network
            ###########################
            gen.zero_grad()
 
            if not prediction_mode:
              noise = Variable(torch.randn(batch_size, noise_dim, 1, 1, device=device))
 
            if use_obgan:
                fake, ba_spread, al_spread, bl_spread, weight_matrices, _, fake_fts = gen(noise, spreads=True, debug=True)
                if prediction_mode:
                  fake = torch.cat((noise, fake), dim=3)
                targets = []
    
                # weight_matrices: array of 10x10x49 matrices
                # goal: 
                #  1. output an array of size |weight_matrices| where each element has all matrices repeated along the first dimension, 9 times
                #  2. output an array of size |weight_matrices| where each element has shifted the input matrix 9 times (-1, -2, -3, -4, ... etc.) along the first dimension
                total_wdiv_loss = 0
                for weight_index in range(len(weight_matrices)):
                  feature_size = weight_matrices[weight_index].shape[0]
                  input = weight_matrices[weight_index].repeat((feature_size - 1, 1, 1))
                  total_fts = feature_size * (feature_size - 1)
                  target = weight_matrices[weight_index].repeat_interleave(feature_size - 1, 0)
                  flattened_size = weight_matrices[weight_index].shape[1] * weight_matrices[weight_index].shape[2]
    
                  # both matrices are 90 x 10 x 49 (L * L-1 x L x T - 1)
                  kl_in = F.log_softmax(input.reshape((total_fts, flattened_size)), dim=1)
                  kl_targ = F.softmax(target.reshape((total_fts, flattened_size)), dim=1)
                  total_wdiv_loss += weight_div_criterion(kl_in, kl_targ)
    
                # calculate order book loss
                ba_loss = ba_criterion(ext_sigmoid(ba_spread, 5), target_ba)
                al_loss = al_criterion(ext_sigmoid(al_spread, 5), target_al_bl)
                bl_loss = bl_criterion(ext_sigmoid(bl_spread, 5), target_al_bl)
    
                output = dis(fake).view(-1)
                
                wgan_loss = -output.mean()
    
                errG = wgan_loss + al_loss + bl_loss + ba_loss - weight_div_lambda * total_wdiv_loss # + (1 - wgan_dis_weight) * wgan_fts_loss 
            else:
                fake = gen(noise, spreads=True, debug=True)
                if prediction_mode:
                  fake = torch.cat((noise, fake), dim=3)
                output = dis(fake).view(-1)
                if use_dcgan_loss:
                  label.fill_(real_label)
                  wgan_loss = dc_loss(output, label)
                else:
                  wgan_loss = -output.mean()
                errG = wgan_loss
            
            errG.backward()

            optimizerG.step()
 
            # Output training stats
            if iters % 500 == 0:
              if use_dcgan_loss:
                print('[%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                    % (iters, errD.item(), errG.item()))
              else:
                print('[%d]\tLoss_D: %.4f\tLoss_G: %.4f\tWDist: %.4f'
                    % (iters, errD.item(), errG.item(), wdistance.item()))
 
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item()) 
            G_gan_loss.append(wgan_loss.item())
 
            if use_obgan:
                G_ba_loss.append(ba_loss.item())
                G_al_loss.append(al_loss.item())
                G_bl_loss.append(bl_loss.item())
                G_wdiv_loss.append(total_wdiv_loss)
        iters += 1

    # After epoch:
    if prediction_mode:
      total_loss = 0
      price_loss = 0
      volume_loss = 0

      price_indices = torch.arange(0, P) * 2
      volume_indices = torch.arange(0, P) * 2 + 1

      gen.eval()

      with torch.no_grad():
        for idx, real_samples in enumerate(test_loader):
          base_rs = real_samples[0].to(device)
          pred_rs = real_samples[1].to(device)
          pred_fake = gen(base_rs)
          total_loss += mse_loss(pred_fake, pred_rs)
          price_loss += mse_loss(pred_fake[:, :, :, price_indices], pred_rs[:, :, :, price_indices])
          volume_loss += mse_loss(pred_fake[:, :, :, volume_indices], pred_rs[:, :, :, volume_indices])

      mean_mse_loss = total_loss.item() / len(test_loader)
      price_mse_loss = price_loss.item() / len(test_loader)
      volume_mse_loss = volume_loss.item() / len(test_loader)
      
      print('[%d]\tMeanMSE: %.4f\tPriceMSE: %.4f\tVolumeMSE: %.4f'
                    % (j + 1, mean_mse_loss, price_mse_loss, volume_mse_loss))

      test_mean_mse_losses.append(mean_mse_loss)
      test_price_mse_losses.append(price_mse_loss)
      test_volume_mse_losses.append(volume_mse_loss)
    
    gen.train()
    # Save model
    torch.save({
        'generator': gen.state_dict(),
        'discriminator': dis.state_dict(),
        'opt_gen': optimizerG.state_dict(),
        'opt_dis': optimizerD.state_dict(),
        'loss_generator': G_losses,
        'loss_discriminator': D_losses,
        'loss_gan': G_gan_loss,
        'loss_bid_ask': G_ba_loss,
        'loss_ask_level': G_al_loss,
        'loss_bid_level': G_bl_loss,
        'loss_wdiv': G_wdiv_loss,
        'iters': iters,
        'mean_mse_losses': test_mean_mse_losses,
        'price_mse_losses': test_price_mse_losses,
        'volume_mse_losses': test_volume_mse_losses,
    }, gdrive_path + "Models/" + model_suffix + ".ptx")

torch.save({
    'generator': gen.state_dict(),
    'discriminator': dis.state_dict(),
    'opt_gen': optimizerG.state_dict(),
    'opt_dis': optimizerD.state_dict(),
    'loss_generator': G_losses,
    'loss_discriminator': D_losses,
    'loss_gan': G_gan_loss,
    'loss_bid_ask': G_ba_loss,
    'loss_ask_level': G_al_loss,
    'loss_bid_level'  : G_bl_loss,
    'loss_wdiv': G_wdiv_loss, 
    'iters': iters,
    'mean_mse_losses': test_mean_mse_losses,
    'price_mse_losses': test_price_mse_losses,
    'volume_mse_losses': test_volume_mse_losses,
}, gdrive_path + "Models/" + model_suffix + ".ptx")