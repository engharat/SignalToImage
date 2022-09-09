################################# Various import #######################################

from __future__ import print_function, division
import cv2
import csv
import os.path
import os
from os import path
from csv import writer
from csv import reader
import argparse
import pathlib
from tqdm import tqdm
import os
import torch
import PIL
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
warnings.filterwarnings("ignore")
from csv import reader
import random
import torchvision.transforms.functional as TF
import glob
from torch.utils.data import SubsetRandomSampler
import numpy as np
################################ Dataset ########################
################################ Dataset ########################

class MultiImgDataset2(Dataset):

    def __init__(self, base_folder, transform=None,N=12):
        #self.norm = norm
        self.num_labels = len(os.listdir(base_folder))
        labels_list = os.listdir(base_folder)

        self.transform = transform
        self.base_folder = base_folder
        self.N = N

        self.images_paths = []
        self.labels = []
        partial_images_paths  = []
        for label in labels_list:
            partial_images_paths = [ f.path for f in os.scandir(os.path.join(base_folder,label)) if f.is_dir() ]
            self.labels += [int(label) for elem in partial_images_paths]
            self.images_paths += partial_images_paths
        for img,label in list(zip(self.images_paths,self.labels)):
            if len(os.listdir(img)) < 6: #ATTENZIONE!!!
                print("ATTENZIONE!!:"+str(img))
                self.images_paths.remove(img)
                self.labels.remove(label)
        self.txt_mode = False if os.path.exists(os.path.join(self.images_paths[1],"Sensor"+str(1)+".png")) else True

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, i):
            images_list = []
            for n in range(1,self.N+1):
                try:
                    frame_file = os.path.join(self.images_paths[i],"Sensor"+str(n)+".png") if not self.txt_mode else os.path.join(self.images_paths[i],"Sensor"+str(n)+".txt")
                except:
                    import pdb; pdb.set_trace()
                if not self.txt_mode:
                    frame = PIL.Image.open(frame_file).convert('RGB')
                    if self.transform:
                        frame = self.transform(frame) #ricorda di passare sempre il transform perchè così il PIL viene trasformato in tensor!
                else:
                    frame = np.loadtxt(frame_file)
                    frame = frame[0:800]
                    frame = torch.from_numpy(frame)
                    frame = frame[None,:]
                images_list.append(frame)
            label = self.labels[i]
            return images_list, label


class MultiImgDataset(Dataset):

    def __init__(self, base_folder, transform=None,N=6,label1="normal",label2="retrofitted"):
        #self.norm = norm
        self.label1 = label1
        self.label2 = label2
        self.transform = transform
        self.base_folder = base_folder
        self.N = N

        self.images_paths = []
        for n in range(self.N):
             self.images_paths.append(glob.glob(base_folder + "/acc"+str(n)+"/**/*.png", recursive = True))

    def __len__(self):
        return len(self.images_paths[0])

    def __getitem__(self, i):
            images_list = []
            for n in range(self.N):
                
                frame_file = self.images_paths[n][i]
                frame = PIL.Image.open(frame_file).convert('RGB')
                if self.transform:
                    frame = self.transform(frame) #ricorda di passare sempre il transform perchè così il PIL viene trasformato in tensor!
                images_list.append(frame)
            if self.label1 in frame_file:
                label = 0
            elif self.label2 in frame_file:
                label = 1
            else:
                raise ValueError('file path does not belong to label1 or label2.')
            return images_list, label




class MultiSignalDataset(Dataset):

    def __init__(self, base_folder, csv_file, N=12):
        df = pd.read_csv(args.csvpath, skipinitialspace=True)

        #self.norm = norm
        self.num_labels = df['Tag'].max() - df['Tag'].min() - 1
        labels_list = [i for i in range(df['Tag'].min(),df['Tag'].max() +1)] 
        self.base_folder = base_folder
        self.N = N

        self.images_paths = []
        self.labels = []
        partial_images_paths  = []
        for label in labels_list:
            partial_images_paths = [ f.path for f in os.scandir(os.path.join(base_folder,label)) if f.is_dir() ]
            self.labels += [int(label) for elem in partial_images_paths]
            self.images_paths += partial_images_paths
        for img,label in list(zip(self.images_paths,self.labels)):
            if len(os.listdir(img)) < 6:
                self.images_paths.remove(img)
                self.labels.remove(label)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, i):
            images_list = []
            for n in range(1,self.N+1):
                try:
                    frame_file = os.path.join(self.images_paths[i],"Sensor"+str(n)+".png")
                except:
                    import pdb; pdb.set_trace()
                frame = PIL.Image.open(frame_file).convert('RGB')
                if self.transform:
                    frame = self.transform(frame) #ricorda di passare sempre il transform perchè così il PIL viene trasformato in tensor!
                images_list.append(frame)
            label = self.labels[i]
            return images_list, label

#df2 = pd.read_csv(os.path.join(args.source,parameter))


def get_train_val_sampler(mydset, percent=0.9, limit=None):
    ## define our indices 
    num_train = len(mydset)
    indices = list(range(num_train))
    split = int(num_train * percent)

    # Random, non-contiguous split
    train_idx = np.random.choice(indices, size=split, replace=False)
    val_idx = list(set(indices) - set(train_idx))
    
    if limit:
        train_idx = train_idx[:limit]
    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]

    ## define our samplers -- we use a SubsetRandomSampler because it will return
    ## a random subset of the split defined by the given indices without replace
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    return train_sampler, val_sampler


