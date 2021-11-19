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

train_list=[]
test_list=[]



for i in range(len(train_test)):
    src = './images/'+images[1][i]
    print(i)
    if train_test[1][i] ==1:
        #train image
        train_list.append(images[1][i])
        
    else:
        #test image
        test_list.append(images[1][i])


train_list = np.asarray(train_list)
test_list = np.asarray(test_list)
with open('train_list.txt', 'w') as fp:
    for i in train_list:
        fp.write(str(i))
        fp.write('\n')
with open('test_list.txt', 'w') as fp:
    for i in test_list:
        fp.write(str(i))
        fp.write('\n')
#np.savetxt('train_list.npy',train_list,delimiter=",")
#np.savetxt('test_list.npy',test_list,delimiter=",")