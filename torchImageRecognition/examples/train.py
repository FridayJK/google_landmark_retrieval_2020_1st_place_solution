import math, re, os
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

from efficientnet_pytorch import EfficientNet

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.python.util.tf_inspect import Parameter

from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import GroupKFold
import pickle
import time


import sys
# print(os.getcwd())
sys.path.append(os.getcwd())
from torchImageRecognition.utils import onnx_conv
from configure import get_arguments

EFF_MODELS = ['efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6']

#configure
args = get_arguments()
NET_ID = args.net_id
INPUT_SIZE = args.input_size
EMB_SIZE = args.embdding_size
NUM_CLASS = args.num_class

class ArcMarginProduct_v2(nn.Module):
    def __init__(self, in_features, NUM_CLASS):
        super(ArcMarginProduct_v2, self).__init__()
        self.NUM_CLASS = NUM_CLASS
        self.w = Parameter(torch.FloatTensor(NUM_CLASS, in_features))
        nn.init.xavier_uniform_(self.w)

    def forward(self, x):
        x = F.linear(F.normalize(x), F.normalize(self.w)) #cosine
        return x


# def ArcMarginProduct_v2(class_num):
class adaCosLoss(torch.nn.Module):
    def __init__(self, class_num):
        super(adaCosLoss, self).__init__()
        self.class_num = class_num
        self.adacos_s = torch.Tensor(np.sqrt(np.array(2)) * np.log(np.array(class_num-1)))
        self.m = 0.1
    def forward(self, labels, logits, mode="train"):
        mask = F.one_hot(labels.to(torch.int64), num_classes=self.class_num)
        theta = torch.acos(torch.clamp(logits, min=-1.0 + 1e-7, max=1.0 - 1e-7))
        with torch.no_grad():
            B_avg = torch.where(mask==1, torch.zeros_like(logits), torch.exp(self.adacos_s * logits))
            B_avg = torch.mean(torch.mean(B_avg, dim=1))
            theta_med = torch.median(theta[mask==1])
            self.adacos_s = torch.log(B_avg)/torch.cos(torch.min(math.pi/4, theta_med))
        output = self.adacos_s*logits

        return output


class efficientEmbNet(nn.Module):
    def __init__(self, NET_ID, EMB_SIZE, NUM_CLASS, re_train=False, pre_trained_model=""):
        super(efficientEmbNet, self).__init__()
        # model = EfficientNet.from_pretrained(EFF_MODELS[NET_ID], num_classes=EMB_SIZE)  #for redefine head, using dif pooling ..
        # self.base_net = nn.Sequential(*list(model.children())[:-4])
        # self.fc1 = nn.Linear(1000, EMB_SIZE, bias=True)
        self.base_net = EfficientNet.from_pretrained(EFF_MODELS[NET_ID], num_classes=EMB_SIZE)
        self.arcMargin = ArcMarginProduct_v2(EMB_SIZE,NUM_CLASS)
        if(re_train):
            dict_model = torch.load(pre_trained_model)
            self.base_net.load_state_dict(dict_model)
        else:
            params = list(self.base_net.named_parameters())
            nn.init.xavier_uniform_(params[-2][1].data)
            nn.init.zeros_(params[-1][1].data)
        
    def forward(self, x):
        x = self.base_net(x)   
        x = self.arcMargin(x)
        return x

#data loader------------------------------------------------------------------------------------------------------------------------
normalize = transforms.Normalize(mean=[], std=[])
preprocess = transforms.Compose([transforms.ToTensor(),
    normalize])
def default_loader(path):
    img = Image.open(path)
    img = img.resize((512,512))
    img_tensor = preprocess(img)

    return img_tensor

class trainset(Dataset):
    def __init__(self, root_path, data_list, loader = default_loader):
        with open(data_list, "rt") as f:
            self.data_list = f.readlines
        self.root_path = root_path
        self.data_loader = loader

    def __getitem__(self, index):
        filename, label = self.data_list[index].split(" ")
        img_tensor = self.data_loader(filename)
        target = label

        return img_tensor, target
    
    def __len__(self):
        return len(self.data_list)

def train(args):
    #config
    NET_ID = args.NET_ID
    INPUT_SIZE = args.INPUT_SIZE
    EMB_SIZE = args.embdding_size
    NUM_CLASS = args.NUM_CLASS

    emb_net = efficientEmbNet(NET_ID, EMB_SIZE, NUM_CLASS)

    # for name_, params_ in emb_net.named_parameters():
    #     print(name_)
    #     print(params_)
    # eff_model.cuda()
    # eff_model.eval()
    #to onnx
    # onnx_name = "efficientnet-b0.onnx"
    # onnx_conv.eff_conv2onnx(eff_model, onnx_name, 9)

    batch_per_cpu = args.batch_per_gpu
    epochs = args.epochs
    lr = args.lr

    data_path = args.data_path
    data_list = args.data_list
    data_list_val = args.data_list_val
    model_save_path = args.model_save_path
    #data process
    train_data = trainset(data_path, data_list)
    train_loader = DataLoader(train_data, batch_size=batch_per_cpu, shuffle=True, num_workers=8)
    val_data = trainset(data_path, data_list_val)
    val_loader = DataLoader(val_data, batch_size=batch_per_cpu, shuffle=True, num_workers=8)
    data_loder = [train_loader, val_loader]
    #train scheduler
    adacos_loss = adaCosLoss(NUM_CLASS)
    optimizer = optim.SGD(emb_net.parameters(), lr=0.01, momentum=0.9)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)
    #train iter
    device = torch.device("cuda:0")
    emb_net.to(device)
    mode_list = ["train", "val"]
    
    for epoch in range(epochs):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("epoch:{}, time:{}".format(epoch, current_time))
        
        loss_list = [0, 0]
        for i, mod in enumerate(mode_list):
            if(mod == "train"):
                emb_net.train()
            else:
                emb_net.eval()
            for datas, labels in tqdm(data_loder[mod]):
                datas = datas.to(device)

                optimizer.zero_grad()
                output=emb_net(datas)
                
                loss = adacos_loss(labels, output)
                loss_list[mod] += loss.item()
                if(mod == "train"):
                    loss.backward()
                    optimizer.step()
        
        print(" train_loss:{}, val_loss:{}".format(loss_list[0]/len(train_loader), loss_list[1]/len(train_loader)))

    return 0

def test():
    return 0


if __name__ == '__main__':

    if(args.work_mode == "train"):
        train(args)
    elif(args.work_mode == "test"):
        test()
