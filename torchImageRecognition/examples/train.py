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

from efficientnet_pytorch import EfficientNet

from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import GroupKFold
import pickle
import time

# print(os.getcwd())
sys.path.append(os.getcwd())
from torchImageRecognition.utils import onnx_conv, utilsEMB
from configure import get_arguments

EFF_MODELS = ['efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6']
EFF_MODEL_NAMES = ["efficientnet-b0-355c32eb.pth",  "efficientnet-b2-8bb594d6.pth",  "efficientnet-b4-6ed6700e.pth",  "efficientnet-b6-c76e70fd.pth",
"efficientnet-b1-f1951068.pth",  "efficientnet-b3-5fb5a3c3.pth",  "efficientnet-b5-b6417697.pth"  "efficientnet-b7-dcc49843.pth"]
#configure
args = get_arguments()
NET_ID = args.net_id
INPUT_SIZE = args.input_size
EMB_SIZE = args.embdding_size
NUM_CLASS = args.num_class
PRE_TRAIN_WEIGHT_PATH = args.pre_trained_weights_path
ROOT_PATH = args.root_path

device = torch.device("cuda:0")

class adaCos(torch.nn.Module):
    def __init__(self, class_num, init_weights = None):
        super(adaCos, self).__init__()
        self.class_num = class_num
        self.emb_size = EMB_SIZE
        self.adacos_s = math.sqrt(2) * math.log(class_num - 1)
        self.pi = torch.Tensor([math.pi]).to(device)
        self.m = 0.1
        self.w = Parameter(torch.FloatTensor(class_num, self.emb_size))
        if(init_weights == None):
            nn.init.xavier_uniform_(self.w)

    def forward(self, labels, input, mode="train"):
        # dot product
        logits = F.linear(F.normalize(input), F.normalize(self.w))

        mask = F.one_hot(labels.to(torch.int64), num_classes=self.class_num).to(device)
        theta = torch.acos(torch.clamp(logits, min=-1.0 + 1e-7, max=1.0 - 1e-7))
        with torch.no_grad():
            B_avg = torch.where(mask==1, torch.zeros_like(logits), torch.exp(self.adacos_s * logits))
            B_avg = torch.mean(torch.sum(B_avg, dim=1))
            theta_med = torch.median(theta[mask==1])
            self.adacos_s = torch.log(B_avg)/torch.cos(torch.min(self.pi/4, theta_med))
        logit = self.adacos_s*logits

        return logit

class efficientEmbNet(nn.Module):
    def __init__(self, NET_ID, EMB_SIZE, re_train=False, pre_trained_model_path=""):
        super(efficientEmbNet, self).__init__()
        self.base_net = EfficientNet.from_pretrained(EFF_MODELS[NET_ID], weights_path= os.path.join(ROOT_PATH, PRE_TRAIN_WEIGHT_PATH, EFF_MODEL_NAMES[NET_ID]), num_classes=EMB_SIZE)
        if(re_train):
            dict_model = torch.load(pre_trained_model_path)
            self.base_net.load_state_dict(dict_model)
        
    def forward(self, x):
        feature = self.base_net(x)
        return feature

#data loader------------------------------------------------------------------------------------------------------------------------
normalize = transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
preprocess = transforms.Compose([transforms.ToTensor(), normalize])
def default_loader(path):
    img = Image.open(path)
    img = img.resize((512,512))
    img_tensor = preprocess(img)

    return img_tensor

class trainset(Dataset):
    def __init__(self, root_path, data_list, loader = default_loader):
        with open(os.path.join(root_path, data_list), "rt") as f:
            self.data_list = f.readlines()
        self.root_path = root_path
        self.data_loader = loader

    def __getitem__(self, index):
        filename, label = self.data_list[index].split(" ")
        img_tensor = self.data_loader(os.path.join(self.root_path, filename))
        target = int(label.strip())

        return img_tensor, target
    
    def __len__(self):
        return len(self.data_list)

def train(args):

    emb_net = efficientEmbNet(NET_ID, EMB_SIZE)
    metric_fc = adaCos(NUM_CLASS)
    criterion = nn.CrossEntropyLoss()

    emb_net.to(device)
    metric_fc.to(device)
    criterion.to(device)

    # for name_, params_ in emb_net.named_parameters():
    #     print(name_)
    #     print(params_)
    # eff_model.cuda()
    # eff_model.eval()
    #to onnx
    # onnx_name = "efficientnet-b0.onnx"
    # onnx_conv.eff_conv2onnx(eff_model, onnx_name, 9)

    batch_per_cpu = args.batch_per_gpu
    data_list = args.data_list
    data_list_val = args.data_list_val
    model_save_path = args.model_save_path
    #data process
    train_data = trainset(ROOT_PATH + "train_rar", data_list)
    train_loader = DataLoader(train_data, batch_size=batch_per_cpu, shuffle=True, num_workers=8)
    val_data = trainset(ROOT_PATH + "train_rar", data_list_val)
    val_loader = DataLoader(val_data, batch_size=batch_per_cpu, shuffle=True, num_workers=8)
    data_loder = [train_loader, val_loader]
    #train scheduler----------------------------------------------------------------------------------------------
    # optimizer = optim.SGD(emb_net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.SGD(emb_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)

    #train--------------------------------------------------------------------------------------------------------
    log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'val_loss'])

    mode_list = ["train", "val"]
    losses_train =  utilsEMB.AverageMeter()
    losses_val =  utilsEMB.AverageMeter()
    losses = [losses_train, losses_val]
    
    for epoch in range(args.epochs):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("epoch:{}, time:{}".format(epoch, current_time))
        scheduler.step()
        for i, mod in enumerate(mode_list):
            if(mod == "train"):
                emb_net.train()
                metric_fc.train()
            else:
                emb_net.eval()
                metric_fc.eval()
            losses[i].reset()
            for datas, labels in tqdm(data_loder[i]):
                datas  = datas.to(device)
                labels = labels.to(device)

                features = emb_net(datas)
                logits = metric_fc(labels, features)
                loss   = criterion(logits, labels)

                losses[i].update(loss.item(), datas.shape[0])
                print("{} loss {}".format(mode_list[i], loss.item()))
                
                if(mod == "train"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        print(" train_loss:{}, val_loss:{}".format(losses[0].avg, losses[1].avg))
        tmp = pd.Series([
            epoch,
            scheduler.get_lr()[0],
            losses[0].avg,
            losses[1].avg,
        ], index=['epoch', 'lr', 'loss', 'val_loss'])
        log = log.append(tmp)
        log.to_csv(ROOT_PATH + '/train_log/log_%s.csv' %time.strftime("%Y-%m-%d#%H:%M:%S"), index=False)
        
    return 0

def test():
    return 0


if __name__ == '__main__':

    if(args.work_mode == "train"):
        train(args)
    elif(args.work_mode == "test"):
        test()
