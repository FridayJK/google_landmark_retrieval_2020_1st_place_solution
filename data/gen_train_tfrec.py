import math, re, os
import tensorflow as tf
import glob
import pandas as pd
from collections import Counter
import numpy as np

train_16fold = pd.read_csv('./data/dataTrain.csv')
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

# list_csv = pd.read_csv('./data/dataTrain.csv')
list_csv = pd.read_csv('./data/dataVal.csv')
landmark_ids = list_csv['clean_landmark_id']
images = list_csv['images']
print("image num:{}".format(len(images)))
#source jpg
# FOLDERNAME = 'v2clean_tfrecord_train1'
FOLDERNAME = 'v2clean_tfrecord_valid1'
DATA_ROOT_PATH = '/workspace/mnt/storage/zhangjunkang/zjk3/data/GLDv2/'+FOLDERNAME+'/'
os.makedirs(DATA_ROOT_PATH,exist_ok=True)
#dist tfrec
DATA_PART = "train_rar"
IMG_ROOT_PATH = '/workspace/mnt/storage/zhangjunkang/zjk3/data/GLDv2/'+DATA_PART+'/'
i=0
for filename in images:
    img_path = IMG_ROOT_PATH + filename[0] + "/" + filename[1] + "/" + filename[2] + "/" + filename + ".jpg"
    img = open(img_path, 'rb').read()
    feature = {
        "_bits":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
        "_class":tf.train.Feature(int64_list=tf.train.Int64List(value=[landmark_ids[i]])),
        "_class_trans":tf.train.Feature(int64_list=tf.train.Int64List(value=[landmark_id_transe_tab[landmark_ids[i]]])),
        '_id':tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(filename)]))
    }
    tfrecord_file = DATA_ROOT_PATH + filename + ".tfrec"
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    i=i+1
    if(i%10000==0):
        print("{} imgs processed!".format(i))