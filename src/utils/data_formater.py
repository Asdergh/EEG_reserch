import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import soundfile as sf
import random as rd
import cv2
import torch as th

from librosa.feature.inverse import mel_to_stft
from librosa import griffinlim
from seaborn import lineplot
from tensorflow.keras.datasets import cifar100

plt.style.use("dark_background")
_, axis = plt.subplots(nrows=3, ncols=2)

images = "C:\\Users\\1\\Desktop\\data (1)\\data\\eeg\\image\\data.pkl"
mels = pd.read_pickle(images)
train_mels = th.Tensor(mels["x_train"])
test_mels = th.Tensor(mels["x_test"])

(train_images_all, train_labels), (test_images_all, test_labels) = cifar100.load_data()
train_images = []
test_images = []
train_labels = np.squeeze(train_labels)
test_labels = np.squeeze(test_labels)
for cll in np.unique(train_labels):

    cll_tr_img = cv2.resize(
        train_images_all[train_labels==cll][np.random.randint(0, 100)],
        (128, 128)
    )
    cll_ts_img = cv2.resize(
         test_images_all[test_labels==cll][np.random.randint(0, 100)],
         (128, 128)
    )


    train_images.append(cll_tr_img)
    test_images.append(cll_ts_img)


train_images = th.Tensor(np.asarray(train_images))
test_images = th.Tensor(np.asarray(test_images))
th.save((train_images, train_mels[:100]), "train_data.pt")
th.save((test_images, test_mels[:100]), "test_data.pt")
print("DATA WAS MOOVED IN .pt FILES!!!")





    
    

    



    




    


