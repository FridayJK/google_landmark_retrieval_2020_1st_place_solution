import math, re, os
import tensorflow as tf
import glob
import pandas as pd
from collections import Counter
import numpy as np

train_16fold = pd.read_csv('./data/dataTrain_stage1.csv')
landmarkIdCounter = dict(Counter(train_16fold['clean_landmark_id']))
train_16fold['counts'] = [landmarkIdCounter[x] for x in train_16fold['clean_landmark_id']]
countIdList = []
for key in sorted(landmarkIdCounter):
    countIdList.append(landmarkIdCounter[key])

landmark_id_transe_tab = {}
i=0
for key in sorted(landmarkIdCounter):
    landmark_id_transe_tab[key] = i
    i = i+1

list_csv = pd.read_csv('./data/dataTrain_stage1.csv')
# list_csv = pd.read_csv('./dataVal_stage1.csv')
landmark_ids = list_csv['clean_landmark_id']
images = list_csv['images']
print("image num:{}".format(len(images)))
i=0
j=0
train_list = []
val_list = []
with open('train_list.txt','w') as f, open('val_list.txt','w') as f1:
    for filename in images:
        img_path = filename[0] + "/" + filename[1] + "/" + filename[2] + "/" + filename + ".jpg"
        new_label = landmark_id_transe_tab[landmark_ids[i]]
        if(i>=1 and landmark_ids[i]==landmark_ids[i-1]):
            j = j+1
            if(j>4 and i!=len(images)-1 and landmark_ids[i]!=landmark_ids[i+1]):
                f1.write(img_path+' '+str(new_label)+'\n')
        else:
            j=0

        # train_list.append([img_path, new_label])
        f.write(img_path+' '+str(new_label)+'\n')
        i=i+1
        if(i%10000==0):
            print("{} imgs processed!".format(i))

# with open('train_list.txt','w') as f:
#     for list in train_list:
#         f.write(str(list)+'\n')