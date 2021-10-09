import math, re, os, sys, time
# os.environ['WORLD_SIZE'] = '1'
# os.environ['RANK'] = '1'

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, dataloader

import torch.distributed as dist

from efficientnet_pytorch import EfficientNet

# print(os.getcwd())
sys.path.append(os.getcwd())
from torchImageRecognition.utils import onnx_conv, utilsEMB, lossWeights, cutOut
from torchImageRecognition.datasets import data_loader
from configure import get_arguments

EFF_MODELS = ['efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6']
EFF_MODEL_NAMES = ["efficientnet-b0-355c32eb.pth", "efficientnet-b1-f1951068.pth", "efficientnet-b2-8bb594d6.pth", "efficientnet-b3-5fb5a3c3.pth", "efficientnet-b4-6ed6700e.pth", "efficientnet-b5-b6417697.pth", "efficientnet-b6-c76e70fd.pth"]
#configure
args = get_arguments()
NET_ID = args.net_id
INPUT_SIZE = args.input_size
EMB_SIZE = args.embdding_size
NUM_CLASS = args.num_class
PRE_TRAIN_WEIGHT_PATH = args.pre_trained_weights_path
ROOT_PATH = args.root_path

# 0. set up distributed device
# rank = int(os.environ["RANK"])
# local_rank = int(os.environ["LOCAL_RANK"])
# torch.cuda.set_device(rank % torch.cuda.device_count())
# dist.init_process_group(backend="nccl")
device = torch.device("cuda", args.local_rank)

loss_weights = lossWeights.loss_weight(os.path.join(ROOT_PATH, "train_rar", args.data_list), args.data_argument)

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class ArcMarginProduct_v2(nn.Module):
    def __init__(self, in_features, NUM_CLASS):
        super(ArcMarginProduct_v2, self).__init__()
        self.NUM_CLASS = NUM_CLASS
        self.w = Parameter(torch.FloatTensor(NUM_CLASS, in_features))
        nn.init.xavier_uniform_(self.w)

    def forward(self, x):
        x = F.linear(F.normalize(x), F.normalize(self.w)) #cosine
        return x

class adaCos(torch.nn.Module):
    def __init__(self, class_num):
        super(adaCos, self).__init__()
        self.class_num = class_num
        self.emb_size = EMB_SIZE
        self.adacos_s = math.sqrt(2) * math.log(class_num - 1)
        self.pi = torch.Tensor([math.pi]).to(device)
        self.m = 0.1

    def forward(self, labels, logits, mode="train"):
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
        self.arcMargin = ArcMarginProduct_v2(EMB_SIZE,NUM_CLASS)
        if(re_train):
            dict_model = torch.load(pre_trained_model_path)
            self.base_net.load_state_dict(dict_model)
        
    def forward(self, x):
        feature = self.base_net(x)
        logits  = self.arcMargin(feature)
        return logits

#data loader------------------------------------------------------------------------------------------------------------------------
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean=args.mean, std=args.std)
preprocess = transforms.Compose([transforms.ToTensor(), normalize, transforms.RandomHorizontalFlip(0.5)])

preprocess_aug1 = transforms.Compose([transforms.ToTensor(), normalize, transforms.RandomHorizontalFlip(0.5)])
preprocess_aug1.transforms.append(transforms.RandomResizedCrop([512,], scale=(0.6,1.0)))
preprocess_aug1.transforms.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3))
preprocess_aug1.transforms.append(cutOut.Cutout(1, 100))

def default_loader(path, mark_aug=False):
    img = Image.open(path)
    if(not mark_aug):
        img = img.resize((INPUT_SIZE[0],INPUT_SIZE[1]))
        img_tensor = preprocess(img)
    else:
        img = img.resize((INPUT_SIZE[0],INPUT_SIZE[1]))
        img_tensor = preprocess_aug1(img)

    return img_tensor

#train-------------------------------------------------------------------------------------------------------------------------------
def train(args):

    # dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    dist.init_process_group(backend='nccl', init_method='env://')
    print("local_rank:{}".format(args.local_rank))
    #
    emb_net   = efficientEmbNet(NET_ID, EMB_SIZE)
    metric_fc = adaCos(NUM_CLASS)
    if(args.use_LossWeight):
        print("use loss weight")
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    emb_net.to(device)
    # emb_net    = nn.parallel.DistributedDataParallel(emb_net)
    emb_net    = nn.parallel.DistributedDataParallel(emb_net, device_ids=[args.local_rank], output_device=args.local_rank)
    # metric_fc  = metric_fc.to(device)
    metric_fc  = metric_fc.to(device)
    criterion  = criterion.to(device)

    #train scheduler----------------------------------------------------------------------------------------------
    optimizer = optim.SGD(emb_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if(args.scheduler == "stepLR"):
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.stepLR_step, gamma=0.2)
    elif(args.scheduler == "multiStepLR"):
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.multiStepLR_step,gamma=0.1)
    elif(args.scheduler == "CosineAnnealing"):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    cudnn.benchmark = True

    data_list       = args.data_list
    data_list_val   = args.data_list_val
    model_save_path = args.model_save_path
    #data process
    train_data    = data_loader.trainset(ROOT_PATH + "train_rar", data_list, loader=default_loader)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader  = DataLoader(train_data, batch_size=args.batch_size, num_workers=8, pin_memory=True, sampler=train_sampler)
    val_data      = data_loader.trainset(ROOT_PATH + "train_rar", data_list_val, loader=default_loader)
    val_sampler   = torch.utils.data.distributed.DistributedSampler(val_data)
    val_loader    = DataLoader(val_data, batch_size=args.batch_size, num_workers=8, pin_memory=True, sampler=val_sampler)
    
    #train--------------------------------------------------------------------------------------------------------
    log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'val_loss'])

    mode_list    = ["train", "val"]
    losses_train = utilsEMB.AverageMeter()
    losses_val   = utilsEMB.AverageMeter()
    losses       = [losses_train, losses_val]
    data_loder   = [train_loader, val_loader]
    current_time = time.strftime('%Y:%m:%d:%H:%M:%S', time.localtime(time.time()))
    print("begin time:{}".format(current_time))
    for epoch in range(args.epochs):
        losses[0].reset()
        losses[1].reset()
        data_loder[0].sampler.set_epoch(epoch)
        data_loder[1].sampler.set_epoch(epoch)
        for i, mod in enumerate(mode_list):
            if(mod == "train"):
                emb_net.train()
                metric_fc.train()
            else:
                emb_net.eval()
                metric_fc.eval()
            losses[i].reset()
            for datas, labels in data_loder[i]:
                datas  = datas.to(device)
                labels = labels.to(device)

                features = emb_net(datas)
                logits = metric_fc(labels, features)
                loss   = criterion(logits, labels)

                losses[i].update(loss.item(), datas.shape[0])
                # print("{} loss {}".format(mode_list[i], loss.item()))
                
                if(mod == "train"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        scheduler.step()

        print(" train_loss:{}, val_loss:{}".format(losses[0].avg, losses[1].avg))
        #save model
        save_path = args.model_save_path + EFF_MODELS[args.net_id] + args.train_note
        os.makedirs(save_path, exist_ok=True)
        if(args.local_rank == 0):
            torch.save(emb_net.module.state_dict(), os.path.join(save_path, EFF_MODELS[args.net_id] + "epoch" + str(epoch) + ".pth"))
            # onnx_conv.eff_conv2onnx(emb_net, base_name + ".onnx", 9)

            tmp = pd.Series([
                epoch,
                scheduler.get_last_lr()[0],
                losses[0].avg,
                losses[1].avg,
            ], index=['epoch', 'lr', 'loss', 'val_loss'])
            log = log.append(tmp, ignore_index=True)
            log.to_csv(save_path + '/log_epoch%d_%s.csv' %(epoch, time.strftime("%Y-%m-%d#%H:%M:%S")), index=False)
        
    return 0

def test():
    return 0


if __name__ == '__main__':

    if(args.work_mode == "train"):
        train(args)
    elif(args.work_mode == "test"):
        test()
