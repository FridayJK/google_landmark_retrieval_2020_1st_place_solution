# from google.colab import drive
# drive.mount('/content/gdrive')
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
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()
print("REPLICAS: ", strategy.num_replicas_in_sync)

# 2-----------------------
EFNS = [efn.EfficientNetB0,efn.EfficientNetB1,efn.EfficientNetB2,efn.EfficientNetB3,
        efn.EfficientNetB4,efn.EfficientNetB5,efn.EfficientNetB6,efn.EfficientNetB7]

# 3-----------------------
def get_model_v1(IMAGE_SIZE, NUM_CLASSES, EMB_SIZE, EFF_VER, order=0,weight_path=None):
    class ArcMarginProduct_v2(tf.keras.layers.Layer):
        def __init__(self, num_classes):
            super(ArcMarginProduct_v2, self).__init__()
            self.num_classes= num_classes
        def build(self, input_shape):
            self.w = self.add_variable(
                "weights", shape=[int(input_shape[-1]), self.num_classes])
        def call(self, input):
            cosine = tf.matmul(tf.nn.l2_normalize(input, axis=1), tf.nn.l2_normalize(self.w, axis=0))
            return cosine
    def getefn():
        pretrained_model = EFNS[EFF_VER](weights=None, include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        pretrained_model.trainable = True
        return pretrained_model
    def ArcFaceResNet():
        x= inputs = tf.keras.Input([*IMAGE_SIZE, 3])
        x = getefn()(x)
        x = L.GlobalAveragePooling2D()(x)
        x = L.Dense(EMB_SIZE, activation='swish')(x)
        target = ArcMarginProduct_v2(NUM_CLASSES)(x)
        model = tf.keras.Model(inputs, target)
        model.get_layer('efficientnet-b'+str(EFF_VER))._name='efficientnet-b'+str(EFF_VER)+str(order)
        return model
    model = ArcFaceResNet()
    model.summary()
    if weight_path is not None:
        model.load_weights(weight_path)
    return model

# 4-----------------------
#single model
model = get_model_v1([640,640],203094,512,6,1,'/content/gdrive/My Drive/eff6_640_notclean_0.5_1.1931.hdf5')
_model= tf.keras.Model(inputs= model.input, 
                       outputs =model.get_layer('dense').output)
def export_model_v1(model, outdir):
    @tf.function(input_signature=[
        tf.TensorSpec(
            shape=[None, None, 3],
            dtype=tf.uint8,
            name='input_image')
    ])
    def serving(input_image):
        image = tf.image.resize(input_image, [640,640])
        image -= tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])  # RGB
        image /= tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])  # RGB
        image = tf.reshape(image, [640,640,3])

        outputs = model(image[tf.newaxis])
        features = tf.math.l2_normalize(outputs[0])        
        return {
            'global_descriptor': tf.identity(features, name='global_descriptor')
        }
    tf.saved_model.save(
    obj=model,
    export_dir=outdir,
    signatures={'serving_default': serving})
export_model_v1(_model,'/content/gdrive/My Drive/landmark_export_model/eff6_640_notclean05_11931')

# 5-----------------------
model1 = get_model_v1([640,640],203094,512,7,1,'/content/gdrive/My Drive/eff7_640_notclean_0.5_1.1411.hdf5')
model2 = get_model_v1([640,640],203094,512,6,2,'/content/gdrive/My Drive/eff6_640_notclean_0.5_1.1931.hdf5')
model3 = get_model_v1([640,640],203094,512,7,3,'/content/gdrive/My Drive/landmark/eff7_512_ver2_10293_NotClean_0.5_640/shuffle_weights.epoch44.loss1.5736.valid_loss1.2554.hdf5')
model4 = get_model_v1([512,512],203094,512,7,4,'/content/gdrive/My Drive/eff7_512_ver1_notclean0.5_1.2580.hdf5')
_model= tf.keras.Model(inputs= [model1.input, 
                                model2.input,
                                model3.input,
                                model4.input,
                                ],
                       outputs =[model1.get_layer('dense').output,
                                 model2.get_layer('dense_1').output,
                                 model3.get_layer('dense_2').output,
                                 model4.get_layer('dense_3').output,
                                 ])
def export_model_v1(model, outdir):
    @tf.function(input_signature=[
        tf.TensorSpec(
            shape=[None, None, 3],
            dtype=tf.uint8,
            name='input_image')
    ])
    def serving(input_image):
        image2 = tf.image.resize(input_image, [640,640])
        image2 -= tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])  # RGB
        image2 /= tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])  # RGB
        image2 = tf.reshape(image2, [640,640,3])
        image3 = tf.image.resize(input_image, [512,512])
        image3 -= tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])  # RGB
        image3 /= tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])  # RGB
        image3 = tf.reshape(image3, [512,512,3])
        outputs = model((image2[tf.newaxis],image2[tf.newaxis],image2[tf.newaxis],image3[tf.newaxis]))
        output1 = tf.math.l2_normalize(outputs[0][0])
        output2 = 0.8*tf.math.l2_normalize(outputs[1][0])
        output3 = 0.55*tf.math.l2_normalize(outputs[2][0])
        output4 = 0.5*tf.math.l2_normalize(outputs[3][0])
        features =  tf.concat([output1,output2,output3, output4],axis=-1)
        return {
            'global_descriptor': tf.identity(features, name='global_descriptor')
        }
    tf.saved_model.save(
    obj=model,
    export_dir=outdir,
    signatures={'serving_default': serving})
export_model_v1(_model,'/content/gdrive/My Drive/landmark_export_model/0816_notclean05_640_776_512_7')
