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


#%% method 4: custom layer
class computeMaskedOutput(tf.keras.layers.Layer):
  def __init__(self):
    super(computeMaskedOutput, self).__init__()

  # def build(self, input_shape):
  #   self.kernel = self.add_weight("kernel",
  #                                 shape=[int(input_shape[-1]),
  #                                        self.num_outputs])

  def call(self, input):
    mus_b=[]#tf.zeros([x.shape[0],x.shape[3],2])
    for k in range(input.shape[0]):
        mus=[]#tf.zeros([x.shape[3],2])
        for i in range(input.shape[3]):#for each filter
            fmap = input[k,:,:,i]
            mu = tf.unravel_index(tf.argmax(tf.reshape(fmap,[-1])),fmap.shape)
            mus.append(mu)
            #mus=tf.stack([mu, mu])
        #mus_b.append(mus)
        mus_b.append(mus)
    mus = tf.convert_to_tensor(mus_b)
    
    templates = []
    for j in range(input.shape[0]):
        template = tf.stack([t_p[j,mus[j,i,0],mus[j,i,1],:,:] for i in range(input.shape[3])])
        templates.append(template)
        # selected_templates.append(template)
    templates = tf.convert_to_tensor(templates)
    templates=tf.transpose(templates,perm=[0,2,3,1])
    
    masked = input*tf.dtypes.cast(templates,tf.float32)
    masked = tf.keras.layers.ReLU()(masked)
    
    return masked
#%%
o_my = computeMaskedOutput()(x)
plt.imshow(o_my[0,:,:,0],cmap='gray')