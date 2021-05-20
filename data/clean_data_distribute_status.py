import pandas as pd
from collections import Counter
import numpy as np

train_clean = pd.read_csv('/workspace/mnt/storage/zhangjunkang/gldv2/data/train_labels/train_labels/train_clean.csv')
landmark_ids = train_clean['landmark_id']
images = train_clean['images']

data_count = np.zeros([10000],dtype="int")
for i, img_id_list in enumerate(images):
    img_id_list = img_id_list.split(" ")
    data_count[len(img_id_list)] = data_count[len(img_id_list)]+1
    if(i%10000==0):
        print("Processed {} rows".format(i))

for i in range(100):
    print("sample num>={}' class count {}".format(i*100, data_count[i*100:].sum()))


all_sample = 0
sample_a16 = 0
sample_a30 = 0
sample_a50 = 0
sample_a80 = 0
sample_a100 = 0
for i in range(10000):
    all_sample = all_sample + i*data_count[i]
    if(i>16):
        sample_a16 = sample_a16 + i*data_count[i]
    if(i>30):
        sample_a30 = sample_a30 + i*data_count[i]
    if(i>50):
        sample_a50 = sample_a50 + i*data_count[i]
    if(i>80):
        sample_a80 = sample_a80 + i*data_count[i]
    if(i>100):
        sample_a100 = sample_a100 + i*data_count[i]

print("sample num >16:{}, >30:{}, >50:{}, >80:{}, >100:{}".format(sample_a16, sample_a30, sample_a50, sample_a80, sample_a100))