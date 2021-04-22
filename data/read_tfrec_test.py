import math, re, os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import GroupKFold
import pickle
import time

AUTO = tf.data.experimental.AUTOTUNE
IMAGE_SIZE = [512,512]
EPOCHS = 2000
BATCH_SIZE_PER_TPU = 1
EFF_VER = 7
# EFF_VER = 0
EMB_SIZE=512
BATCH_SIZE = BATCH_SIZE_PER_TPU 
FOLDERNAME = 'v2clean_sample'
# DRIVE_DS_PATH = '/content/gdrive/My Drive/'+FOLDERNAME
DRIVE_DS_PATH = './data/'+FOLDERNAME
os.makedirs(DRIVE_DS_PATH,exist_ok=True)
NUM_CLASSES = 81313

GCS_DS_PATH = "/workspace/mnt/storage/zhangjunkang/zjk3/data/GLDv2"
TRAIN_GCS_PATH = GCS_DS_PATH + '/v2clean_tfrecord_train'
TRAIN_FILENAMES = tf.io.gfile.glob(TRAIN_GCS_PATH + '/*.tfrec')
VALID_GCS_PATH = GCS_DS_PATH + '/v2clean_tfrecord_valid'
VALID_FILENAMES = tf.io.gfile.glob(VALID_GCS_PATH + '/*.tfrec')

def normalize_image(image):
    image -= tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])  # RGB
    image /= tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])  # RGB
    return image
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    # image = normalize_image(image)
    # image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image
def img_aug(image, label):
    img = tf.image.random_flip_left_right(image)
    return img, label
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "_bits": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "_class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
        '_id': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['_bits'])
    label = tf.cast(example['_class'],tf.int32)
    return image, label

def load_dataset(filenames, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order,increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    return dataset

def get_training_dataset():
    dataset = load_dataset(TRAIN_FILENAMES,ordered=False)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.map(img_aug, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset():
    dataset = load_dataset(VALID_FILENAMES,ordered=True)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.batch(4)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

    
# NUM_TRAINING_IMAGES = count_data_items(TRAIN_FILENAMES)
# NUM_VALIDATION_IMAGES = count_data_items(VALID_FILENAMES)
NUM_TRAINING_IMAGES = len(TRAIN_FILENAMES)
NUM_VALIDATION_IMAGES = len(VALID_FILENAMES)
print('Dataset: {} training images'.format(NUM_TRAINING_IMAGES))
print('Dataset: {} validation images'.format(NUM_VALIDATION_IMAGES))

valid_data_iter = iter(get_validation_dataset())

for i in range(5):
    img, labs = next(valid_data_iter)
    plt.imshow(img.numpy().squeeze())
    plt.savefig("jpg_test/"+str(labs.numpy()[0])+".jpg")
    # image = tf.image.encode_jpeg(img.numpy().squeeze())
    # tf.io.gfile.GFile(str(labs.numpy()[0])+".jpg","wb").write(image.eval())


with sess.as_default():
  print(loss.eval())