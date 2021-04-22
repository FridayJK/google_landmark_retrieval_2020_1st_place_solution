import math, re, os
import tensorflow as tf
import glob

FOLDERNAME = 'ctrain_tfrec'
DATA_ROOT_PATH = '/workspace/mnt/storage/zhangjunkang/zjk3/data/GLDv2/'+FOLDERNAME+'/'
os.makedirs(DATA_ROOT_PATH,exist_ok=True)
# index_tfrec_file = "mnt/data/GLDv2/index.tfrec"

images = glob.glob('/mnt/data/GLDv2/index_images/*/*/*/*.jpg')
i=0
for filename in images:
    img = open(filename, 'rb').read()
    # img_id = filename.split('/')[-1].split('.')[0]
    img_id = filename.split('/')[-1]
    feature = {
        "_bits":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
        "_class":tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
        '_id':tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(img_id)]))
    }
    tfrecord_file = DATA_ROOT_PATH + img_id.split('.')[0] + ".tfrec"
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    i=i+1
    if(i%1000==0):
        print("{} imgs processed!".format(i))