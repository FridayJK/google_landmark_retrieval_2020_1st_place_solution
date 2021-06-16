import math, re, os
import glob
import pandas as pd
from collections import Counter
import numpy as np
import multiprocessing
import tensorflow as tf

train_csv = pd.read_csv('/workspace/mnt/storage/zhangjunkang/gldv2/data/train_labels/train_labels/train.csv')
landmark_ids = train_csv['landmark_id']
images = train_csv['id']
print("image num:{}".format(len(images)))
#source jpg
FOLDERNAME = 'tfrecord_train'
# FOLDERNAME = 'v2clean_tfrecord_valid'
TF_ROOT_PATH = '/workspace/mnt/storage/zhangjunkang/gldv1/gldv2/'+FOLDERNAME+'/'
os.makedirs(TF_ROOT_PATH,exist_ok=True)
#dist tfrec
DATA_PART = "train_rar"
IMG_ROOT_PATH = '/workspace/mnt/storage/zhangjunkang/gldv2/data/'+DATA_PART+'/'

# images_list = [img for img in images]
# landmark_ids_list = [ids for ids in landmark_ids]

images_infs = [[img, ids] for img,ids in zip(images, landmark_ids)]

# def main(images_inf):
#     i=0
#     images, landmark_ids = images_inf
#     for filename in images:
#         img_path = IMG_ROOT_PATH + filename[0] + "/" + filename[1] + "/" + filename[2] + "/" + filename + ".jpg"
#         img = open(img_path, 'rb').read()
#         feature = {
#             "_bits":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
#             "_class":tf.train.Feature(int64_list=tf.train.Int64List(value=[landmark_ids[i]])),
#             '_id':tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(filename)]))
#         }
#         tfrecord_file = TF_ROOT_PATH + filename + ".tfrec"
#         with tf.io.TFRecordWriter(tfrecord_file) as writer:
#             example = tf.train.Example(features=tf.train.Features(feature=feature))
#             writer.write(example.SerializeToString())
#         i=i+1
#         if(i%10000==0):
#             print("{} imgs processed!".format(i))

def main(images_inf):
    filename, landmark_ids = images_inf
    img_path = IMG_ROOT_PATH + filename[0] + "/" + filename[1] + "/" + filename[2] + "/" + filename + ".jpg"
    img = open(img_path, 'rb').read()
    feature = {
        "_bits":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
        "_class":tf.train.Feature(int64_list=tf.train.Int64List(value=[landmark_ids])),
        '_id':tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(filename)]))
    }
    tfrecord_file = TF_ROOT_PATH + filename + ".tfrec"
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=8)
    pool.map(main, images_infs)
    # main(images_infs[0])