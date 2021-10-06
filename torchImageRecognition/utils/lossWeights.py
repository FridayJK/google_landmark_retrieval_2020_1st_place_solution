import math, re, os
import numpy as np
import pandas as pd

import torch
from collections import Counter

def loss_weight(list_path):
    # train_fold = pd.read_csv("/workspace/mnt/storage/zhangjunkang/gldv1/gldv2/train_rar/dataTrain_stage1.txt", sep=" ", header=None, names=["images", "clean_landmark_id"])
    train_fold = pd.read_csv(list_path, sep=" ", header=None, names=["images", "clean_landmark_id"])
    landmarkIdCounter = dict(Counter(train_fold['clean_landmark_id']))
    train_fold['counts'] = [landmarkIdCounter[x] for x in train_fold['clean_landmark_id']]
    countIdList = []
    for key in sorted(landmarkIdCounter):
        countIdList.append(landmarkIdCounter[key])
    scaleV = 1/ np.mean(1/(np.log(np.array(train_fold['counts']))))

    lossWeight = torch.tensor([scaleV/(np.log(np.array(countIdList)))], dtype=torch.float32)
    # lossWeight = lossWeight.repeat(16,1)
    return lossWeight
