import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2

data_path = 'data/open'

train_png = sorted(glob(os.path.join(data_path,"train/*.png")))
test_png = sorted(glob(os.path.join(data_path,"test/*.png")))

train_df = pd.read_csv(os.path.join(data_path,'train_df.csv'))

train_labels = train_df["label"]

label_unique = sorted(np.unique(train_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

train_labels = [label_unique[k] for k in train_labels]

def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (384, 384))

    return img

def load_train_test(img1=train_png,img2=test_png):
    train_imgs = []
    test_imgs = []
    for m in tqdm(train_png):
        train_imgs.append(img_load(m))
    for n in tqdm(test_png):
        test_imgs.append(img_load(n))


    return train_imgs, test_imgs

# train_imgs, test_imgs = load_train_test()
