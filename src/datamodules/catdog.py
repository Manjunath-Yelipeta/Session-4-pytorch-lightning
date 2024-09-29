import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from pathlib import Path
from typing import Union
import os
import os
import sys
import stat

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import random_split, SubsetRandomSampler
from torchvision import datasets, transforms, models 
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer
import lightning.pytorch as pl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
import gdown
import zipfile
from zipfile import ZipFile



transform=transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])

class DogDataModule(pl.LightningDataModule):
    
    def __init__(self, transform=transform, batch_size=32):
        super().__init__()
        self.root_dir = "ext/dataset"
        self.transform = transform
        self.batch_size = batch_size

    def setup(self, stage=None):
        file_id = "1-lbK2hPkOeZB-RfwLmeBj3oOSbsWU-ov"
        output_file = "exp/image_data.zip"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file)
        #os.chmod("image_data.zip", stat.S_IRWXO)

#output = "dog-breed-image-dataset_gd_exp2.zip"
#gdown.download(url, output)
        #download_and_extract_archive(zipfile="/workspace/lightning-template-hydra/data/dt1.zip",
                                      #download_root="/workspace/lightning-template-hydra/data/exp",
                                      #remove_finished=True,)
        

        #with zipfile.ZipFile('exp/image_data.zip', 'w') as new_zip:
            #new_zip.write('new.zip', compress_type=zipfile.ZIP_DEFLATED)
        zip_ref = zipfile.ZipFile("exp/image_data.zip", 'r')
        zip_ref.extractall("ext")
        zip_ref.close()
        dataset = datasets.ImageFolder(root=self.root_dir, transform=self.transform)
        n_data = len(dataset)
        n_train = int(0.8 * n_data)
        n_test = n_data - n_train
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
        self.train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = DataLoader(test_dataset, batch_size=self.batch_size)

    def train_dataloader(self):
        return self.train_dataset

    def test_dataloader(self):
        return self.test_dataset
