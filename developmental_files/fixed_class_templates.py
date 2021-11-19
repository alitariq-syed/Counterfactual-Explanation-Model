# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 21:37:49 2020

@author: Ali
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
n=12
k=128
n_classes=10
batch = np.round(np.random.random(32)*9) #batch classes

#%% initialize class map and templates
filters_per_class = tf.math.round(k/n_classes)        
class_map = -1*np.zeros((k),dtype=np.uint8)
count=0
for i in range(n_classes):
    for j in range(int(filters_per_class)):
        class_map[count]= i
        count+=1
        if count == k:
            break

class_templates = np.stack([np.zeros((n,n),dtype=np.uint8) for i in range(k)])
class_templates = np.stack([class_templates for i in range(n_classes)])

class_templates = np.transpose(class_templates,[0,2,3,1])
#class_templates dimensions = class,filters,fmapx,fmapy

# create class_templates for each class
for i in range(k):
    class_templates[class_map[i],:,:,i] = np.ones((n,n))

class_templates = tf.convert_to_tensor(class_templates)
#%%
#
fig, axs = plt.subplots(8,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
#fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
for i in range(32):

    axs[i].imshow(class_templates[1,:,:,i],cmap='gray',vmin=0, vmax=1)#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
    axs[i].axis('off')

#plt.title('templates_1')    
plt.show()
#%% assign class templates for each image in current batch
#input--> class templates, batch classes

#batch_templates = np.stack([class_templates[int(batch[i])]] for i in range(len(batch)))

batch_templates=[]
for i in range(len(batch)):
    batch_templates.append(class_templates[int(batch[i])])
batch_templates = tf.convert_to_tensor(batch_templates)

#%%
#
fig, axs = plt.subplots(32,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
#fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
for i in range(128):

    axs[i].imshow(batch_templates[0,:,:,i],cmap='gray',vmin=0, vmax=1)#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
    axs[i].axis('off')

#plt.title('templates_1')    
plt.show()