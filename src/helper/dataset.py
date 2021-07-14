import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import minmax_scale
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_seed = 54231856
torch.manual_seed(exp_seed)
np.random.seed(exp_seed)

"""
Class that handles reading the data set and normalizes the data
"""
class OrderBookDataSet(Dataset):
    # vpc: values per column
    def __init__(self, dir, transform, T=50, vpc=20, prediction_horizon=0, size=1, set_type="train"):
        self.dir = dir
        self.transform = transform
        self.data = np.loadtxt(self.dir)
        
        if size == -1:
          if set_type == "test":
            self.raw_data = self.data[150000:200000, :]
          else:
            self.raw_data = self.data[:150000, :]
          self.raw_data = self.raw_data.reshape(-1, 2)
          self.raw_data = minmax_scale(self.raw_data, axis=0)
          self.raw_data = self.raw_data.reshape(-1, 40)

        if size > 0:
          self.data = self.data[:40, :].T
          stock_indices = []
          if set_type == "train":
            if size == 1:
              stock_indices = [3454, 9772, 14694, 25413]
            elif size == 2:
              stock_indices = [8533, 20973, 31860, 49291]
            elif size == 4:
              stock_indices = [15242, 40085, 60341, 93777]
          elif set_type == "test":
            if size == 4:
              stock_indices = [3030, 9758, 15704, 22082]
            elif size == 1:
              stock_indices = [5079, 11201, 17166, 23878]
          
          # Normalize each stock seperately to make sure all data is in the same range 
          s1 = self.data[:stock_indices[0]]
          s2 = self.data[stock_indices[0]:stock_indices[1],:]
          s3 = self.data[stock_indices[1]:stock_indices[2],:]
          s4 = self.data[stock_indices[2]:stock_indices[3],:]
          s5 = self.data[stock_indices[3]:,:]
  
          # Step 1: Reshape data to have prices in one column and volumes in the other
          # Step 2: Concatenate all rows to have ALL prices in one column and ALL volumes in the other
          # Step 3: Normalize rows (minmax_scale)
          # Step 4: Reshape data to n x 20 x 2
          # r = int(40 / vpc)
          s1_valid = minmax_scale(s1.reshape((-1, 2)), axis=0).reshape((-1, 40, 1))
          s2_valid = minmax_scale(s2.reshape((-1, 2)), axis=0).reshape((-1, 40, 1))
          s3_valid = minmax_scale(s3.reshape((-1, 2)), axis=0).reshape((-1, 40, 1))
          s4_valid = minmax_scale(s4.reshape((-1, 2)), axis=0).reshape((-1, 40, 1))
          s5_valid = minmax_scale(s5.reshape((-1, 2)), axis=0).reshape((-1, 40, 1))
  
          self.raw_data = np.concatenate((s1_valid, s2_valid, s3_valid, s4_valid, s5_valid), axis=0)
 
        self.data = self.raw_data.reshape((-1, vpc, int(40 / vpc)))
 
        self.vpc = vpc
        self.base_T = T
        self.prediction_horizon = prediction_horizon
        self.T = self.base_T + self.prediction_horizon

    def get_data(self):
      return self.raw_data
 
    def get_sample_shape(self):
        return self.T * int(40 / self.vpc), self.vpc
 
    def __len__(self):
        return len(self.data) - self.T
 
    def __getitem__(self, index):
        if self.prediction_horizon == 0:
          sample = np.concatenate(self.data[index:index + self.T], axis=1)
          return self.transform(sample).float()
        else:
          last_base_index = index + self.base_T
          last_sample_index = last_base_index + self.prediction_horizon
          base_sample = np.concatenate(self.data[index:last_base_index], axis=1)
          ext_sample = np.concatenate(self.data[last_base_index:last_sample_index], axis=1)
          return (self.transform(base_sample).float(), self.transform(ext_sample).float())