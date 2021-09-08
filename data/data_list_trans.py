#landmark id trans for train clean data

import math, re, os
import glob
import pandas as pd
from collections import Counter
import numpy as np

#read csv
train_clean = pd.read_csv('/workspace/mnt/storage/zhangjunkang/gldv1/gldv2/train_labels/train_clean.csv')
landmark_ids = train_clean['landmark_id']
images = train_clean['images']

#trans table dict
# trans_tab = {}
# for i, land_id in enumerate(landmark_ids):
#     trans_tab[str(land_id)] = i

#
root_path = "/workspace/mnt/storage/zhangjunkang/gldv1/gldv2/"
train_list_file = root_path + "data_list/dataTrain_stage1.txt"
val_list_file = root_path + "data_list/dataVal_stage1.txt"
with open(train_list_file, "wt") as f1, open(val_list_file, "wt") as f2:
    for i, img_list in enumerate(images):
        img_list = img_list.split(" ")
        for j, img_name in enumerate(img_list):
            img_path = img_name[0] + "/" + img_name[1] + "/" + img_name[2] + "/" + img_name + ".jpg" + " " + str(i) + "\n"
            if(j==0 and len(img_list)>=4):
                f2.write(img_path)
            else:
                f1.write(img_path)
        if((i+1)%1000 == 0):
            print("{} processed".format(i+1))

print("Done!")