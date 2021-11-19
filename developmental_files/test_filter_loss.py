# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 06:41:01 2020

@author: Ali
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt


#%%
x = tf.convert_to_tensor(np.load('fmaps.npy'))
t_p = tf.convert_to_tensor(np.load('templates.npy'))

x1 = x[0,:,:,0]
plt.imshow(x1,cmap='gray')
x1_np = x1.numpy()

#%% Determining the target category for each filter: We need to assign each filter f with a target category ˆc
# We simply assign the filter f with the category ˆc whose images activate f the most, 

#i.e single-batch output shape = [32, 10, 10, 32]

# pick all feature maps (corresponding to different images) for 1 filter

filter_1 = x[:,:,:,1]#.numpy()

#%%
filter_1_means=[]

for img in range(x.shape[0]):
    filter_1_means.append( tf.reduce_mean(filter_1[img]))
filter_1_means = tf.convert_to_tensor(filter_1_means)

#%% suppose the ground truth of images in batch
gt_classes = tf.convert_to_tensor(tf.keras.utils.to_categorical(np.random.randint(0,10,(32))))

#%%
classes = tf.argmax(gt_classes,axis=1)

#%% class-wise mean activation
filter_1_means_class_wise=np.zeros((10))
class_sums = np.zeros((10))
for i in range(32):
    filter_1_means_class_wise[tf.argmax(gt_classes[i])] += filter_1_means[i]
    class_sums[tf.argmax(gt_classes[i])] +=1 
filter_1_means_class_wise_average = filter_1_means_class_wise/class_sums
filter_class = np.argmax(filter_1_means_class_wise_average)
