# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:02:26 2020

@author: Ali
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#%% creating part templates

#x = nxn feature map; x[i,j] >=0
# suppose n = 14

#%% create sample feature map
n=12
#x = np.zeros((n,n),dtype=np.float32)
x = np.random.rand(n,n)
#plt.plot(x)
#plt.show()


#%% create positive template
t = (0.5)/np.power(n,2) #tao is positive constant
b = 2 #beta is postive constant#con

mu = np.unravel_index(np.argmax(x),x.shape) # strongest activation location

t_p = np.zeros(x.shape)
for i in range(n):
    for j in range(n):
        t_p[i,j] = t*max(1-b*(np.linalg.norm(np.array([i, j]) - mu, ord=1)/n),-1)
plt.imshow(t_p,cmap='gray')
#manhattan distance: numpy.linalg.norm(a - b, ord=1)
#a=np.array([10, 7])
#np.linalg.norm(a - b, ord=1)

#%% create negative template
t_n = np.ones(x.shape)*-t


#%% convert to tensor??

#t_p_tensor = tf.convert_to_tensor(t_p, np.float32)
