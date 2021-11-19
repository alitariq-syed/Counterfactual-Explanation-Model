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
#%%
mus_b=[]
for k in range(x.shape[0]):
    mus=[]
    for i in range(x.shape[3]):#for each filter
        fmap = x[k,:,:,i]
        mu = np.unravel_index(np.argmax(fmap),fmap.shape)
        mus.append(mu)
    mus_b.append(mus)
mus = np.asarray(mus_b)

#%% testing for 1 fmap
# selected_templates = tf.stack(t_p[0,mus[0,0,0],mus[0,0,1],:,:])
# plt.imshow(x1,cmap='gray'),plt.show()
# plt.imshow(selected_templates,cmap='gray'),plt.show()

# selected_templates = tf.dtypes.cast(selected_templates, tf.float32)
# masked = x1*selected_templates
# masked = tf.keras.layers.ReLU()(masked)
# plt.imshow(masked,cmap='gray'),plt.show()

#%% for all fmaps
templates = []
for j in range(x.shape[0]):
    template = tf.stack([t_p[j,mus[j,i,0],mus[j,i,1],:,:] for i in range(x.shape[3])])
    templates.append(template)
    # selected_templates.append(template)
templates = tf.convert_to_tensor(templates)
templates=tf.transpose(templates,perm=[0,2,3,1])

#%%
b=12
ind = 4
plt.imshow(x[b,:,:,ind],cmap='gray'),plt.show()
plt.imshow(templates[b,:,:,ind],cmap='gray'),plt.show()

#selected_templates = tf.dtypes.cast(selected_templates, tf.float32)
masked = x[b,:,:,ind]*tf.dtypes.cast(templates[b,:,:,ind],tf.float32)
masked = tf.keras.layers.ReLU()(masked)
plt.imshow(masked,cmap='gray'),plt.show()
#%% method 2
# from tensorflow.keras.layers import MaxPool2D


# class my_maxpool2d(MaxPool2D):
#     def __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs):
#         super(my_maxpool2d, self).__init__()
        
# o_my = my_maxpool2d(10,10)(x)

#%% method 3
mus_b=[]#tf.zeros([x.shape[0],x.shape[3],2])
for k in range(x.shape[0]):
    mus=[]#tf.zeros([x.shape[3],2])
    for i in range(x.shape[3]):#for each filter
        fmap = x[k,:,:,i]
        mu = tf.unravel_index(tf.argmax(tf.reshape(fmap,[-1])),fmap.shape)
        mus.append(mu)
        #mus=tf.stack([mu, mu])
    #mus_b.append(mus)
    mus_b.append(mus)
mus = tf.convert_to_tensor(mus_b)

#%% method 4: max_pool with argmax
out = tf.nn.max_pool_with_argmax(x, ksize=(x.shape[1],x.shape[2]), strides=(x.shape[1],x.shape[2]), 
                           output_dtype=tf.dtypes.int64, include_batch_in_index=True,padding='SAME')
indices = tf.unravel_index(tf.reshape(out[1],[-1]),x.shape)
#indices returning indices of shape (4,1024) --> 4=(batch,x,y,filters); 1024 = 32*32 (batch*filters)

#The indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes 
#flattened index: (y * width + x) * channels + c
# 800 = (0,2,5,0) = (2*32+5)*32+0

#%%
#selected_templates = t_p[indices]
templates = []
for i in range(x.shape[0]):
    for j in range(x.shape[3]):
        st = indices[1:3,(i*x.shape[3] + j)]
        template = t_p[j,st[0],st[1],:,:]
        templates.append(template)
templates = tf.convert_to_tensor(templates)
templates = tf.stack([templates[i*x.shape[0]:i*x.shape[0]+x.shape[3],:,:] for i in range(x.shape[0])])
templates = tf.transpose(templates,[0,2,3,1])

#%% #the above is still faster
templates = []
for i in range(x.shape[0]):
    st = indices[1:3,i*x.shape[0]:i*x.shape[0]+x.shape[3]]
    template = tf.stack([t_p[0,st[0,j],st[1,j],:,:] for j in range(x.shape[3])])
    templates.append(template)
templates = tf.convert_to_tensor(templates)
templates = tf.transpose(templates,[0,2,3,1])
