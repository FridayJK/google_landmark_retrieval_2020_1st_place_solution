import math, re, os, sys, time
import multiprocessing

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
sys.path.append("/root/google_landmark_retrieval_2020_1st_place_solution")
from torchImageRecognition.utils import onnx_conv, utilsEMB
from torchImageRecognition.datasets import data_loader
from torchImageRecognition.examples.configure import get_arguments
from torchImageRecognition.eval.gldv2.compute_retrieval_metrics import metric_gldv2

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

device = torch.device("cuda", args.local_rank)


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
        # logits  = self.arcMargin(feature)
        return feature

#data loader------------------------------------------------------------------------------------------------------------------------
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean=args.mean, std=args.std)
preprocess = transforms.Compose([transforms.ToTensor(), normalize])
def default_loader(path):
    img = Image.open(path)
    img = img.resize((INPUT_SIZE[0],INPUT_SIZE[1]))
    img_tensor = preprocess(img)

    return img_tensor

#train-------------------------------------------------------------------------------------------------------------------------------
def train(args):
    #
    emb_net   = efficientEmbNet(NET_ID, EMB_SIZE)
    metric_fc = adaCos(NUM_CLASS)
    criterion = nn.CrossEntropyLoss()

    emb_net.to(device)
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
    train_data = data_loader.trainset(ROOT_PATH + "train_rar", data_list, loader=default_loader)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)

    val_data      = data_loader.trainset(ROOT_PATH + "train_rar", data_list_val, loader=default_loader)
    val_loader    = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
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
        for i, mod in enumerate(mode_list):
            j=0
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
                # print("{} loss {}".format(mode_list[i], loss.item()))
                
                if(mod == "train"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # j+=1
                # if(j>100):
                #     break

        scheduler.step()

        print(" train_loss:{}, val_loss:{}".format(losses[0].avg, losses[1].avg))
        #save model
        save_path = args.model_save_path + EFF_MODELS[args.net_id] + args.train_note
        os.makedirs(save_path, exist_ok=True)
        if(args.local_rank == 0):
            torch.save(emb_net.state_dict(), os.path.join(save_path, EFF_MODELS[args.net_id] + "epoch" + str(epoch) + ".pth"))
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

def extract_feat(data_path, data_list, model_path):
    emb_net = efficientEmbNet(NET_ID, EMB_SIZE)
    # emb_net = nn.DataParallel(emb_net)
    emb_net.load_state_dict(torch.load(model_path))
    emb_net.to(device)
    cudnn.benchmark = True

    #--------------------------------
    batch_size = 64
    val_data      = data_loader.testset(data_path, data_list, loader=default_loader)
    val_loader    = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)
    
    emb_net.eval()

    features = np.empty([len(data_list), EMB_SIZE], dtype="float32")
    with torch.no_grad():
        for i, datas in enumerate(val_loader):
            datas   = datas.to(device)
            time1 = time.time()
            feature = emb_net(datas)
            time2 = time.time()
            # print("batch:{}, time:{}".format(batch_size, time2 - time1))
            if(feature.shape[0] == batch_size):
                begin = i*batch_size
                end   = begin+batch_size
                features[begin:end,:] = feature.cpu().numpy()
            else:
                begin = i*batch_size
                end   = begin+feature.shape[0]
                features[begin:end,:] = feature.cpu().numpy()
            # print((i+1)*batch_size)
        
    return features

def test(model_path):
    root_path = "/workspace/mnt/storage/zhangjunkang/zjk3/data/GLDv2/"
    test_csv = pd.read_csv(os.path.join(root_path, "test_labels/retrieval_solution_v2.1.csv"))
    test_ids = test_csv['id']
    test_usage = test_csv['Usage']
    #filter
    images = []
    Usage = []
    for i, filename in enumerate(test_ids):
        if(test_usage[i]!="Ignored"):
            images.append(test_ids[i])
            Usage.append(test_usage[i])
    #test
    test_feat = extract_feat(root_path + "test_rar", images, model_path) ###
    feat_Path = "./result/feat/"+args.train_note
    os.makedirs(feat_Path,exist_ok=True)
    np.save(feat_Path+"/testFeat_"+ os.path.basename(model_path).split(".")[0] + ".npy",test_feat)
    #inde
    index_csv = pd.read_csv(os.path.join(root_path + "index_labels/index.csv"))
    index_ids = index_csv["id"]
    index_feat = extract_feat(root_path + "index_rar", index_ids, model_path)       ###
    np.save(feat_Path+"/indexFeat_" + os.path.basename(model_path).split(".")[0] + ".npy",index_feat)

    # test_feat  = np.load(feat_Path+"/testFeat_"+ os.path.basename(model_path).split(".")[0] + ".npy")
    # index_feat = np.load(feat_Path+"/indexFeat_" + os.path.basename(model_path).split(".")[0] + ".npy")

    #exec retrieval
    test_feat = test_feat/np.linalg.norm(test_feat,2,1)[:,None]
    index_feat = index_feat/np.linalg.norm(index_feat,2,1)[:,None]
    sim_mat = np.inner(test_feat,index_feat)
    sim_mat_list = [x for x in sim_mat]
    pool = multiprocessing.Pool(processes=64)
    sort_idx = pool.map(np.argsort, sim_mat_list)
    sim_mat_sorted = pool.map(np.sort, sim_mat_list)
    pool.close()
    pool.join()
    sort_idx = np.vstack(sort_idx)
    sort_idx = sort_idx[:,::-1]
    topK_index = sort_idx[:,0:100].tolist()

    #save solution topK images
    test_image_id = []
    solution_image_id = []
    index_list_csv = pd.read_csv('/workspace/mnt/storage/zhangjunkang/zjk3/data/GLDv2/index_labels/index.csv')
    index_images = index_list_csv['id']
    for i in range(len(topK_index)):
        solution_image_id.append(" ".join(np.array(index_images[topK_index[i]]).tolist()))

    pred_Path = "./result/prediction/"+args.train_note
    os.makedirs(pred_Path,exist_ok=True)

    predictions = pd.DataFrame({'id':images,'images':solution_image_id,'Usage':Usage})
    predictions.to_csv(pred_Path+"/predictions_"+os.path.basename(model_path).split(".")[0]+".csv",sep=',', index=False)
    #metric
    print("================================================================================================================================================")
    print("***** MAP of {} *****".format(model_path))
    print("================================================================================================================================================")
    metric_gldv2(pred_Path+"/predictions_"+os.path.basename(model_path).split(".")[0]+".csv")

if __name__ == '__main__':

    # if(args.work_mode == "train"):
    #     train(args)
    # elif(args.work_mode == "test"):
    model_root_path = "/workspace/mnt/storage/zhangjunkang/gldv2/model/pytorch/efficientnet-b3test5_net3/"
    model_list = ["efficientnet-b3epoch30.pth","efficientnet-b3epoch35.pth", "efficientnet-b3epoch40.pth", "efficientnet-b3epoch45.pth", "efficientnet-b3epoch49.pth"]
    for model in model_list:
        test(model_root_path + model)
