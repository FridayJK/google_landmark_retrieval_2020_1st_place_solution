import math, re, os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.python.util.tf_inspect import Parameter

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
import torchvision.models as models
import onnx

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

import efficientnet.tfkeras as efn

from efficientnet_pytorch import EfficientNet

from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import GroupKFold
import pickle
import time
import argparse

import sys
# print(os.getcwd())
sys.path.append(os.getcwd())
from torchImageRecognition.utils import onnx_conv

EFF_MODELS = ['efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6']

class ArcMarginProduct_v2(nn.Module):
    def __init__(self, in_features, num_class):
        super(ArcMarginProduct_v2, self).__init__()
        self.num_class = num_class
        self.w = Parameter(torch.FloatTensor(num_class, in_features))
        nn.init.xavier_uniform_(self.w)

    def forward(self, x):
        x = F.linear(F.normalize(x), F.normalize(self.w)) #cosine
        return x


# def ArcMarginProduct_v2(class_num):
class adaCosLoss():
    def __init__(self):
        self.adacos_s = 
        self.pi = 
        self.theta_zero = self.pi/4
        self.m = 0.5
    def get_loss(labels, logits, mode):

        return 


class efficientEmbNet(nn.Module):
    def __init__(self, net_id, emb_size, num_class):
        super(efficientEmbNet, self).__init__()
        self.base_net = EfficientNet.from_pretrained(EFF_MODELS[net_id])
        self.fc1 = nn.Linear(1000, emb_size, bias=True)
        self.arcMargin = ArcMarginProduct_v2(emb_size,num_class)
        
    def forward(self, x):
        x = self.base_net(x)
        #global pooling
        x = self.fc1(x)
        x = self.arcMargin(x)
        return x

    

def train(args):
    #config
    net_id = args.net_id
    input_size = args.input_size
    emb_size = args.embdding_size
    num_class = args.num_class

    eff_model = EfficientNet.from_pretrained(EFF_MODELS[net_id])
    eff_model.cuda()
    eff_model.eval()
    #to onnx
    onnx_name = "efficientnet-b0.onnx"
    onnx_conv.eff_conv2onnx(eff_model, onnx_name, 9)

    batch_per_cpu = args.batch_per_gpu
    epochs = args.epochs
    lr = args.lr

    data_path = args.data_path
    model_save_path = args.model_save_path
    #data process
    
    #train scheduler
    #train iter
    return 0

def test():
    return 0

def get_arguments():
    args = argparse.ArgumentParser()

    args.add_argument('--net_id', type=int, default=0, help="efficientnet id num")
    args.add_argument('--input_size', type=list, default=[512,512], help="input size of net")
    args.add_argument('--embdding_size', type=int, default=512)
    args.add_argument('--num_class', type=int, default=81313)
    args.add_argument('--data_argument', type=bool, default=False)

    args.add_argument('--from_scratch', type=bool, default=False)
    args.add_argument('--pre_trained_weights', type=str, default="")
    args.add_argument('--loss_type', type=str, default="softmax", choices=['softmax', 'adacos'])
    args.add_argument('--scheduler', type=str, default="stepLR", choices=['stepLR', 'multiStepLR'])
    args.add_argument('--lr', type=float, default=1e-4)
    args.add_argument('--epochs', type=int, default=50)
    args.add_argument('--batch_per_gpu', type=int, default=50)

    args.add_argument('--data_path', type=str, default="")
    args.add_argument('--model_save_path', type=str, default="")
    
    args.add_argument('--work_mode', type=str, default="train", choices=['train', 'test'])

    return args.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    if(args.work_mode == "train"):
        train(args)
    elif(args.work_mode == "test"):
        test()
