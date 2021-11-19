# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:39:44 2020

@author: Ali
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = tf.convert_to_tensor(np.load('fmaps.npy'))
t1 = tf.zeros_like(x)

x1 = x#[0,:,:,0]
#plt.imshow(x1,cmap='gray')

target1 = t1#[0,:,:,0]
#plt.imshow(target1,cmap='gray')

#%%
loss = tf.reduce_mean(tf.keras.losses.MAE(target1,x1))