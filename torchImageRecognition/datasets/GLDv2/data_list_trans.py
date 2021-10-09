#landmark id trans for train clean data

import math, re, os
import glob
import pandas as pd
import random
from collections import Counter
import numpy as np

#read csv
train_clean = pd.read_csv('/workspace/mnt/storage/zhangjunkang/gldv1/gldv2/train_labels/train_clean.csv')
landmark_ids = train_clean['landmark_id']
images = train_clean['images']


#
def ori_trans():
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

def data_augment_trans(low_bound=15, num_argue=20):
    root_path = "/workspace/mnt/storage/zhangjunkang/gldv2/data/train_labels/"
    train_list_file = root_path + "dataTrain_stage1_augment" + str(low_bound) + "_" + str(num_argue) + "_list.txt"
    val_list_file = root_path + "dataVal_stage1_augment" + str(low_bound) + "_" + str(num_argue) + "_list.txt"
    with open(train_list_file, "wt") as f1, open(val_list_file, "wt") as f2:
        for i, img_list in enumerate(images):
            img_list = img_list.split(" ")
            for j, img_name in enumerate(img_list):
                img_path = img_name[0] + "/" + img_name[1] + "/" + img_name[2] + "/" + img_name + ".jpg" + " " + str(i) + " 0" + "\n"
                if(j==0 and len(img_list)>=4):
                    f2.write(img_path)
                else:
                    f1.write(img_path)
                #augment
                if(j==(len(img_list)-1)):
                    if(len(img_list)<4):
                        aug_num = num_argue - len(img_list)
                        for aug_idx in range(aug_num):
                            idx = aug_idx%(len(img_list))
                            aug_name = img_list[idx]
                            img_path = aug_name[0] + "/" + aug_name[1] + "/" + aug_name[2] + "/" + aug_name + ".jpg" + " " + str(i) + " 1" + "\n"
                            f1.write(img_path)
                    elif(len(img_list)<num_argue):
                        aug_num = num_argue - len(img_list) + 1
                        for aug_idx in range(aug_num):
                            idx = aug_idx%(len(img_list) - 1)
                            aug_name = img_list[idx + 1]
                            img_path = aug_name[0] + "/" + aug_name[1] + "/" + aug_name[2] + "/" + aug_name + ".jpg" + " " + str(i) + " 1" + "\n"
                            f1.write(img_path)


            if((i+1)%1000 == 0):
                print("{} processed".format(i+1))
    print("Done!")

if __name__ == "__main__":
    data_augment_trans()
