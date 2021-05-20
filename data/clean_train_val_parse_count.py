
import pandas as pd
from collections import Counter
import random
import numpy as np

train_clean = pd.read_csv('/workspace/mnt/storage/zhangjunkang/gldv2/data/train_labels/train_labels/train_clean.csv')
landmark_ids = train_clean['landmark_id']
images = train_clean['images']

COUNT_LIMITED = 16

train_list_land_id = []
train_list_img_id = []
val_list_land_id = []
val_list_img_id = []
for i, img_id_list in enumerate(images):
    img_id_list = img_id_list.split(" ")
    if(len(img_id_list) >= 4):
        val_list_land_id.append(landmark_ids[i])
        val_idx = random.randint(0, len(img_id_list)-1)
        val_list_img_id.append(img_id_list[val_idx])
        img_id_list.remove(img_id_list[val_idx])
    img_id_list2 = random.sample(img_id_list, min(len(img_id_list), COUNT_LIMITED))
    for j, img_id in enumerate(img_id_list2):
        train_list_land_id.append(landmark_ids[i])
        train_list_img_id.append(img_id_list[j])
    if(i%10000==0):
        print("Processed {} rows".format(i))

print("DataSet {} training imgs".format(len(train_list_land_id)))
print("DataSet {} val imgs".format(len(val_list_land_id)))
dataTrain = pd.DataFrame({'clean_landmark_id':train_list_land_id,'images':train_list_img_id})
dataTrain.to_csv("/workspace/mnt/storage/zhangjunkang/gldv2/data/dataTrain"+ str(COUNT_LIMITED) +"_stage1.csv",sep=',')

dataVal = pd.DataFrame({'clean_landmark_id':val_list_land_id,'images':val_list_img_id})
dataVal.to_csv("/workspace/mnt/storage/zhangjunkang/gldv2/data/dataVal"+ str(COUNT_LIMITED) +"_stage1.csv",sep=',')
