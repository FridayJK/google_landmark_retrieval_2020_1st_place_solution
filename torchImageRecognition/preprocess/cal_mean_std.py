import math, re, os, sys
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import math

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, dataloader

sys.path.append(os.getcwd())
from torchImageRecognition.utils import onnx_conv, utilsEMB
from torchImageRecognition.dataset import data_loader

preprocess = transforms.Compose([transforms.ToTensor()])
def cal_loader(path):
    img = Image.open(path)
    img = img.resize((512,512))
    img_tensor = preprocess(img)

    return img_tensor

data_list_val = "dataVal_stage1.txt"
ROOT_PATH     = "/workspace/mnt/storage/zhangjunkang/gldv1/gldv2/train_rar/"
val_data = data_loader.trainset(ROOT_PATH, data_list_val, loader=cal_loader)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True, num_workers=8)


#
psum    = torch.tensor([0.,0.,0.], dtype=torch.float64)
psum_sq = torch.tensor([0.,0.,0.], dtype=torch.float64)
for datas, labels in tqdm(val_loader):
    psum    += datas.sum(axis = [0,2,3])
    psum_sq += (datas**2).sum(axis = [0,2,3])

count = val_data.__len__() * 512 * 512

t_mean = psum/count
t_var  = (psum_sq/count) - (t_mean**2)
t_std  = torch.sqrt(t_var)

print("mean:"+ str(t_mean))
print("std:"+ str(t_std))

