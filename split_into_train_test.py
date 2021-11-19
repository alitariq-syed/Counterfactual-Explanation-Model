# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:27:44 2020

@author: Ali
"""

import numpy as np
import pandas as pd
import os

os.chdir('G:\CUB_200_2011\CUB_200_2011')
os.listdir()



images = pd.read_csv('images.txt', sep=" ", header=None)#np.loadtxt(fname = 'images.txt')
train_test = pd.read_csv('train_test_split.txt', sep=" ", header=None)

train_path = './train_test_split/train/'
test_path = './train_test_split/test/'



from shutil import copyfile
for i in range(len(train_test)):
    src = './images/'+images[1][i]
    print(i)
    if train_test[1][i] ==1:
        #train image
        dst_path = os.path.join(train_path,images[1][i].split('/')[0]+'/')
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst = os.path.join(train_path,images[1][i])    
        copyfile(src, dst)
    else:
        #test image
        dst_path = os.path.join(test_path+images[1][i].split('/')[0]+'/')
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst = os.path.join(test_path+images[1][i])
        copyfile(src, dst)