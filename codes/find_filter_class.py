# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:31:54 2020

@author: Ali
"""

import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm 
from tensorflow.keras.layers import GlobalAveragePooling2D
#%%
def find_filter_class(model, train_gen):
    #assumption: its a standrd model, not interpretable model
    num_filters = model.output[1].shape[3]
    num_classes = model.output[0].shape[1]
    
    class_activation_sums = np.zeros((num_filters,num_classes)) # sum of mean activations per class for each filter... then choose argmax as filter category
    class_img_count = np.zeros((num_classes)) # keep track of number of image samples seen for each class
 
    
    batches=math.ceil(train_gen.n/train_gen.batch_size)
    train_gen.reset()
    #for step,(x_batch_train, y_batch_train) in enumerate(dataset):
    with tqdm(total=batches) as progBar:
        for step in range(batches):
          x_batch_test, y_batch_test = next(train_gen)
          
          predictions,fmaps = model(x_batch_test, training=False)
          
    
          gt_argmax = tf.argmax(y_batch_test,1,output_type=tf.dtypes.int32)
          classes = tf.math.bincount(
              gt_argmax, weights=None, minlength=num_classes, maxlength=None, dtype=tf.dtypes.float32)
        
          class_img_count += classes


    
          # for f in range(num_filters): #number of filters
          #       filter_1 = fmaps[:,:,:,f]
                
          #       for img in range(fmaps.shape[0]):#number of images/batch size
          #           class_activation_sums[f,gt_argmax[img]] = class_activation_sums[f,gt_argmax[img]]+tf.reduce_mean(filter_1[img])
          
            
          #mean_fmap_activations = tf.reduce_mean(tf.reduce_mean(fmaps,axis=1),axis=1)
          
          
          mean_fmap_activations = GlobalAveragePooling2D()(fmaps)
          for img in range(fmaps.shape[0]):#number of images/batch size
              class_activation_sums[:,gt_argmax[img]] = class_activation_sums[:,gt_argmax[img]]+mean_fmap_activations[img,:]
          progBar.update()

    
    return class_activation_sums, class_img_count
    
    
    
    
    
    
    
    
    
    
    