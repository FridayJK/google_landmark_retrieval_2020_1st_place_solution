# !pip install -q efficientnet
# 1-----------------------
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
import sys

BATCH = 32
EMB_SIZE = 512
NUM_CLASSES = 81313
IMAGE_SIZE = [512,512]
EFF_VER = 7
EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6,efn.EfficientNetB7]


def normalize_image(image):
    image -= tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])  # RGB
    image /= tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])  # RGB
    return image
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = normalize_image(image)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

class ArcMarginProduct_v2(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(ArcMarginProduct_v2, self).__init__()
        self.num_classes= num_classes
    def build(self, input_shape):
        self.w = self.add_variable(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
    def call(self, input):
        cosine = tf.matmul(tf.nn.l2_normalize(input, axis=1), tf.nn.l2_normalize(self.w, axis=0))
        # cosine = tf.matmul(input, self.w)
        return cosine
def getefn():
    pretrained_model = EFNS[EFF_VER](weights=None, include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    # pretrained_model = EFNS[EFF_VER](weights='./data/GLDv2_models/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    pretrained_model.trainable = True
    return pretrained_model
def ArcFaceResNet():
    x= inputs = tf.keras.Input([*IMAGE_SIZE, 3], name='input_image')
    x = getefn()(x)
    x = L.GlobalAveragePooling2D()(x)
    # x = L.Dense(EMB_SIZE, activation='swish')(x)
    # x = L.Dense(EMB_SIZE, activation=None)(x)
    # x = L.Dense(EMB_SIZE, activation=tf.nn.sigmoid)(x)
    x = L.Dense(EMB_SIZE, activation=None, use_bias=False)(x)
    # target = ArcMarginProduct_v2(NUM_CLASSES)(x)
    return tf.keras.Model(inputs, x)

model = ArcFaceResNet()
# model.load_weights('./data/GLDv2_models/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',by_name=True)
# model.load_weights('./models/GLDv2_models2/2080_m3init_e2_softmax_dense3/weights.epoch03.loss0.2655.valid_loss0.9343.hdf5',by_name=True)
# model.load_weights('./models/GLDv2_models2/2080_m3init_e2_softmax_dense3_lr/weights.epoch01.loss0.0546.valid_loss0.8465.hdf5',by_name=True)
# model.load_weights('./models/GLDv2_models2/A100_m3init_e2_softmax_dense1/weights.epoch14.loss0.0031.valid_loss1.7363.hdf5',by_name=True)
model.load_weights('./models/GLDv2_models_M7/A100_M7init_e3_softmax1/weights.epoch03.loss0.3954.valid_loss1.1176.hdf5',by_name=True)
model.summary()

def extract_feat(data_set, imgList = []):
    data_root_path = '/workspace/mnt/storage/zhangjunkang/zjk3/data/GLDv2/'+data_set+'_rar/'
    data_list_csv = pd.read_csv('/workspace/mnt/storage/zhangjunkang/zjk3/data/GLDv2/' + data_set + '_labels/'+ data_set+ '.csv')
    if(len(imgList)):
        images = imgList
    else:
        images = data_list_csv['id']
    index_feat = np.empty([len(images),EMB_SIZE],dtype="float32")
    input_batch = tf.Variable(np.empty((BATCH, 512,512,3), dtype=np.float32))
    for i, filename in enumerate(images):
        img_path = data_root_path + filename[0] + "/" + filename[1] + "/" + filename[2] + "/" + filename + ".jpg"
        img = open(img_path, 'rb').read()
        input = decode_image(img)
        input_batch[i%BATCH].assign(input)
        if(i==len(images)-1):
            output = model(input_batch)
            output = tf.nn.l2_normalize(output, axis=1)
            index_feat[(i-i%BATCH):(i+1),:] = output.numpy()[0:(i%BATCH+1)]
            continue
        if((i+1)%BATCH!=0):
            continue
        # output = model(tf.reshape(input,[1,512,512,3]))
        output = model(input_batch)
        output = tf.nn.l2_normalize(output, axis=1)
        index_feat[(i-BATCH+1):(i+1),:] = output.numpy()
        if((i+1)%(BATCH*100)==0):
            print("Processed {} rows".format(i))
    return index_feat


if __name__ == '__main__':
    DATA_SET = "test"
    DATA_ROOT_PATH = '/workspace/mnt/storage/zhangjunkang/zjk3/data/GLDv2/'+DATA_SET+'_rar/'
    data_list_csv = pd.read_csv('/workspace/mnt/storage/zhangjunkang/zjk3/data/GLDv2/' + DATA_SET + '_labels/retrieval_solution_v2.1.csv')
    test_id = data_list_csv['id']
    test_usage = data_list_csv['Usage']
    images = []
    Usage = []
    for i, filename in enumerate(test_id):
        if(test_usage[i]!="Ignored"):
            images.append(test_id[i])
            Usage.append(test_usage[i])
    test_feat = extract_feat("test", images)
    # np.save("testFeatModel3_A100loss0.0031.valid_loss1.7363_rmIgnored.npy",test_feat)
    np.save("testFeatM7_A100epoch03.loss0.3954.valid_loss1.1176_rmIgnored.npy",test_feat)
    # test_feat = np.load("test_feat_model3init_rmIgnored.npy")
    # test_feat = np.load("testFeatModel3_valid_loss0.8465_rmIgnored.npy")

    index_feat = extract_feat("index")
    # np.save("indexFeatModel3_A100loss0.0031.valid_loss1.7363.npy",index_feat)
    np.save("indexFeatModel7_A100epoch03.loss0.3954.valid_loss1.1176.npy",index_feat)
    # index_feat = np.load("index_model3init_feat.npy")
    # index_feat = np.load("indexFeatModel3_valid_loss0.8465.npy")

    sim_mat = np.inner(test_feat,index_feat)
    sort_idx = np.argsort(sim_mat,axis = 1)
    sort_idx = sort_idx[:,::-1]
    topK_index = sort_idx[:,0:100].tolist()

    #save solution topK images
    test_image_id = []
    solution_image_id = []
    index_list_csv = pd.read_csv('/workspace/mnt/storage/zhangjunkang/zjk3/data/GLDv2/index_labels/index.csv')
    index_images = index_list_csv['id']
    for i in range(len(topK_index)):
        solution_image_id.append(" ".join(np.array(index_images[topK_index[i]]).tolist()))


    predictions = pd.DataFrame({'id':images,'images':solution_image_id,'Usage':Usage})
    # predictions.to_csv("./result/predictions_Model3_A100loss0.0031.valid_loss1.7363.csv",sep=',', index=False)
    predictions.to_csv("./result/predictions_Model7_A100epoch03.loss0.3954.valid_loss1.1176.csv",sep=',', index=False)