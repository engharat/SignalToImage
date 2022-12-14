
import os.path
import os
from os import path
from csv import writer
from csv import reader
import pathlib
from torch.serialization import save
from tqdm import tqdm
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")
from csv import reader
import random
import torchvision.transforms.functional as TF
import glob
import kazane
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import scipy.fft
from vmdpy import VMD 
import matplotlib.pyplot as plt  

class MinMaxScaler3D(MinMaxScaler):

    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=X.shape)

class Standard3D(StandardScaler):

    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=X.shape)

    def transform(self, X):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().transform(x,copy=None), newshape=X.shape)

################################ Dataset ########################
select_list = ['aBD11Az','aBD17Ay','aBD17Az','aBD17Cz','aBD23Ay','aBD23Az']
class KW51(Dataset):

    def __init__(self, base_folder="~/Downloads/traindata_csv/Train_folder_traindata/",substract=False,max_seq_len=32768,decimate_factor=0,scaler=None,data_aug=False,fft=False,file=None):
        base_folder = os.path.expanduser(base_folder)
        self.substract = substract
        self.data_aug = data_aug
        self.noise_var = 0.01
        self.data_paths = glob.glob(base_folder + "/**/*.csv", recursive = True)
        self.datas=[]
        self.max_seq_len = max_seq_len
        self.decimate_factor = decimate_factor
        if self.decimate_factor > 0:
            self.decimater = kazane.Decimate(self.decimate_factor)
        max_length = 0
        self.scaler = StandardScaler() if scaler is None else scaler
        print("LOADING AND PROCESSING DATA...")

        for i in tqdm(range(len(self.data_paths))):
            df = pd.read_csv(self.data_paths[i])
            df = df[select_list]
            data =  torch.from_numpy(np.nan_to_num(df.astype(np.float32).to_numpy()))
            data = data[0:self.max_seq_len, :]
            if self.decimate_factor > 0:
                data = self.decimater(data.T).T
            if self.substract:
                initial_values = data[0, :].clone()
                data -= torch.roll(data, 1, 0)
                data[0, :] = initial_values

            if data.shape[0]>max_length : max_length = data.shape[0]
            self.datas.append(data)
        indexes = [i for i,elem in enumerate(self.datas) if elem.shape[0] < max_length] #ho ottenuto cos?? alcuni elementi che erano pi?? corti e danno problemi al batch
        for index in sorted(indexes, reverse=True): #li rimuovo
            del self.datas[index]
        self.datas = torch.stack([elem for elem in self.datas])
        self.n_samples,self.seq_len, self.n_features = self.datas.shape

        self.datas = self.datas.view(-1,self.n_features)
        #### 0 mean 1 variance transformation
        self.datas = torch.from_numpy(self.scaler.fit_transform(self.datas.numpy())) if scaler is None else torch.from_numpy(self.scaler.transform(self.datas.numpy()))
        self.datas = self.datas.view(self.n_samples,self.seq_len,self.n_features)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.datas[i,:,:] if not self.data_aug else self.datas[i,:,:] + (self.noise_var**0.5)*torch.randn(self.datas[i,:,:].shape)
