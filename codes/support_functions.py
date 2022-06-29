# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:52:41 2020

@author: Ali
"""
import numpy as np
import tensorflow as tf
import cv2

#%%

def get_heatmap_only(only_heatmap,img):
    original_heatmap = cv2.resize(only_heatmap[0].numpy(), (img.shape[1], img.shape[0]))    
    original_heatmap = (original_heatmap - np.min(original_heatmap)) / (original_heatmap.max() - original_heatmap.min())
    original_heatmap = (original_heatmap * 255).astype("uint8")
    
    
    mask = tf.cast(original_heatmap > 2*np.mean(original_heatmap), "float32")  

    img[:,:,0] = img[:,:,0]*mask
    img[:,:,1] = img[:,:,1]*mask
    img[:,:,2] = img[:,:,2]*mask
    return img
def get_heatmap_only_batch(only_heatmap,img):
    for i in range(len(only_heatmap)):
        original_heatmap = cv2.resize(only_heatmap[i].numpy(), (img.shape[1], img.shape[2]))    
        original_heatmap = (original_heatmap - np.min(original_heatmap)) / (original_heatmap.max() - original_heatmap.min())
        original_heatmap = (original_heatmap * 255).astype("uint8")
        
        
        mask = tf.cast(original_heatmap > 2*np.mean(original_heatmap), "float32")  
    
        img[i,:,:,0] = img[i,:,:,0]*mask
        img[i,:,:,1] = img[i,:,:,1]*mask
        img[i,:,:,2] = img[i,:,:,2]*mask
    return img

#for VGG preprocessing only
def restore_original_image_from_array(x, data_format='channels_last'):
    mean = [103.939, 116.779, 123.68]
    x = x.copy()
    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]

    return x.astype('uint8')