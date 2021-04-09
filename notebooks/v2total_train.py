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
AUTO = tf.data.experimental.AUTOTUNE
IMAGE_SIZE = [512,512]
EPOCHS = 2000
NOT_CLEAN_WEIGHT = 0.5
EMB_DIM=512
BATCH_SIZE_PER_TPU = 4
EFF_VER = 7
BATCH_SIZE = BATCH_SIZE_PER_TPU * strategy.num_replicas_in_sync
FOLDERNAME = 'v2total_sample'
DRIVE_DS_PATH = '/content/gdrive/My Drive/landmark/'+FOLDERNAME
os.makedirs(DRIVE_DS_PATH,exist_ok=True)
NUM_CLASSES = 203094
EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3,
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6,efn.EfficientNetB7]

# 3-----------------------
train_total_16fold = pd.read_csv('/content/gdrive/My Drive/landmark/train_total_16fold.csv')
train_16fold = pd.read_csv('/content/gdrive/My Drive/landmark/train_16fold.csv')
cleanToTotal = tf.constant(sorted(list(set(train_16fold['landmark_id']))))
from collections import Counter
landmarkIdCounter = dict(Counter(train_total_16fold['landmark_id']))
train_total_16fold['counts'] = [landmarkIdCounter[x] for x in train_total_16fold['landmark_id']]
countIdList=[]
for key in sorted(landmarkIdCounter):
    countIdList.append(landmarkIdCounter[key])
isCleanList = list(train_total_16fold['isClean'])
countsList = list(train_total_16fold['counts'])
isCleanLogCountWeightList = []
for i in range(len(isCleanList)):
    if isCleanList[i] == 0:
        isCleanLogCountWeightList.append(NOT_CLEAN_WEIGHT * (1/np.log(countsList[i]+1)))
    else:
        isCleanLogCountWeightList.append(1 * (1/np.log(countsList[i]+1)))
scaleV = (1/np.mean(isCleanLogCountWeightList))
lossWeight =np.array(scaleV/np.log(np.array(countIdList)+1))
lossWeight = tf.constant(lossWeight)
lossWeight = tf.tile(tf.expand_dims(lossWeight,0),tf.constant([BATCH_SIZE_PER_TPU,1]))

# 4-----------------------
#GCS_DS_PATH = 'GCS private bucket path'
TRAIN_GCS_PATH = GCS_DS_PATH + '/v2total_tfrecord_train'
TRAIN_FILENAMES = tf.io.gfile.glob(TRAIN_GCS_PATH + '/*.tfrec')
VALID_GCS_PATH = GCS_DS_PATH + '/v2clean_tfrecord_valid'
VALID_FILENAMES = tf.io.gfile.glob(VALID_GCS_PATH + '/*.tfrec')

# 5-----------------------
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

def img_aug(image, label, isClean):
    img = tf.image.random_flip_left_right(image)
    return img, label, isClean

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "_bits": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "_class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
        '_id': tf.io.FixedLenFeature([], tf.string),
        '_isClean': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['_bits'])
    label = tf.cast(example['_class'],tf.int32)
    isClean = tf.cast(example['_isClean'], tf.int32)
    return image, label, isClean

def valid_read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "_bits": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "_class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
        '_id': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['_bits'])
    label = tf.cast(example['_class'],tf.int32)
    return image, cleanToTotal[label], tf.constant([0])

def load_dataset(filenames, train=True,ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order,increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    if train==True:
        dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        dataset = dataset.map(valid_read_labeled_tfrecord, num_parallel_calls=AUTO)
    return dataset

def get_training_dataset():
    dataset = load_dataset(TRAIN_FILENAMES,train=True,ordered=False)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.map(img_aug, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset():
    dataset = load_dataset(VALID_FILENAMES,train=False,ordered=True)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)
    
NUM_TRAINING_IMAGES = count_data_items(TRAIN_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALID_FILENAMES)
print('Dataset: {} training images'.format(NUM_TRAINING_IMAGES))
print('Dataset: {} validation images'.format(NUM_VALIDATION_IMAGES))

# 6-----------------------
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
    x= inputs = tf.keras.Input([*IMAGE_SIZE, 3], name='input_image')
    x = getefn()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(EMB_DIM, activation='swish')(x)
    target = ArcMarginProduct_v2(NUM_CLASSES)(x)
    return tf.keras.Model(inputs, target)

#references
#https://arxiv.org/abs/1905.00292
#https://github.com/taekwan-lee/adacos-tensorflow/blob/master/adacos.py
class adacosLoss:
    def __init__(self):
        self.adacos_s = tf.math.sqrt(2.0) * tf.math.log(tf.cast(NUM_CLASSES - 1,tf.float32))
        self.pi =  tf.constant(3.14159265358979323846)
        self.theta_zero = self.pi/4
        self.m = 0.5
    def getLoss(self, labels, logits, cleans, mode):
        mask = tf.one_hot(tf.cast(labels, tf.int32), depth = NUM_CLASSES)
        theta = tf.math.acos(tf.clip_by_value(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        B_avg =tf.where(mask==1,tf.zeros_like(logits), tf.math.exp(self.adacos_s * logits))
        B_avg = tf.reduce_mean(tf.reduce_sum(B_avg, axis=1), name='B_avg')
        B_avg = tf.stop_gradient(B_avg)
        theta_class = tf.gather_nd(theta, tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1), name='theta_class')
        theta_med = tfp.stats.percentile(theta_class, q=50)
        theta_med = tf.stop_gradient(theta_med)
        self.adacos_s=(tf.math.log(B_avg) / tf.cos(tf.minimum(self.theta_zero, theta_med)))
        output = tf.multiply(self.adacos_s, logits, name='adacos_logits')        
        cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
        if mode=='train':
            isCleanValues = tf.cast(tf.gather_nd(tf.constant([[NOT_CLEAN_WEIGHT],[1.0]]), tf.stack([cleans,tf.tile(tf.constant([0]),tf.shape(cleans))], axis=1)), tf.float32)
            weightValues = tf.cast(tf.gather_nd(lossWeight, tf.stack([tf.range(BATCH_SIZE_PER_TPU),labels], axis=1)),tf.float32)
            loss = cce(labels, output, sample_weight = tf.multiply(isCleanValues, weightValues))
        else :
            loss = cce(labels, output)
        return loss

# 7-----------------------
with strategy.scope():
model = ArcFaceResNet()
optimizer = tf.keras.optimizers.SGD(1e-3, momentum=0.9, decay = 1e-5)
train_loss = tf.keras.metrics.Sum()
valid_loss = tf.keras.metrics.Sum()
def loss_fn(labels, predictions, cleans, mode='train'):
    _adacosLoss = adacosLoss()
    per_example_loss = _adacosLoss.getLoss(labels, predictions, cleans, mode)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size= BATCH_SIZE)
model.summary()

# 8-----------------------
STEPS_PER_TPU_CALL = NUM_TRAINING_IMAGES // BATCH_SIZE // 8
VALIDATION_STEPS_PER_TPU_CALL = NUM_VALIDATION_IMAGES // BATCH_SIZE
@tf.function
def train_step(data_iter):
    def train_step_fn(images, labels, cleans):
        with tf.GradientTape() as tape:
            cosine = model(images)
            loss = loss_fn(labels, cosine, cleans)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        #update metrics
        train_loss.update_state(loss)
    #this loop runs on the TPU
    for _ in tf.range(STEPS_PER_TPU_CALL):
        strategy.run(train_step_fn, next(data_iter))
@tf.function
def valid_step(data_iter):
    def valid_step_fn(images, labels, cleans):
        probabilities = model(images, training=False)
        loss = loss_fn(labels, probabilities, cleans, 'valid')
        # update metrics
        valid_loss.update_state(loss)
    # this loop runs on the TPU
    for _ in tf.range(VALIDATION_STEPS_PER_TPU_CALL):
        strategy.run(valid_step_fn, next(data_iter))

# 9-----------------------
from collections import namedtuple
start_time = epoch_start_time = time.time()
train_dist_ds = strategy.experimental_distribute_dataset(get_training_dataset())
valid_dist_ds = strategy.experimental_distribute_dataset(get_validation_dataset())
print("Training steps per epoch:", STEPS_PER_EPOCH, "in increments of", STEPS_PER_TPU_CALL)
epoch = START_EPOCH
train_data_iter = iter(train_dist_ds) # the training data iterator is repeated and it is not reset
                                      # for each validation run (same as model.fit)
valid_data_iter = iter(valid_dist_ds)
while True:
    train_step(train_data_iter)
    print('|', end='', flush=True)
    valid_step(valid_data_iter)
    print('=', end='', flush=True)
    trainLossV = train_loss.result().numpy()/STEPS_PER_TPU_CALL
    print('\nEPOCH {:d}/{:d}'.format(epoch+1, EPOCHS))
    print('loss: {:0.4f}'.format(trainLossV),
          'valid_loss : {:0.4f} '.format(valid_loss.result().numpy() / VALIDATION_STEPS_PER_TPU_CALL),
          flush=True)
    model.save_weights(os.path.join(DRIVE_DS_PATH, 'weights.epoch{:02d}.loss{:0.4f}.valid_loss{:0.4f}.hdf5').format(epoch+1, trainLossV,valid_loss.result().numpy() /VALIDATION_STEPS_PER_TPU_CALL))
    epoch += 1
    train_loss.reset_states()
    valid_loss.reset_states()
    if epoch >= EPOCHS:
        break

    