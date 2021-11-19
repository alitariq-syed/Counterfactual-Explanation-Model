# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:39:40 2020

@author: Ali
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#%%
#scores1 = np.array((range(20)))-10.0
#scores2 = 10.0-np.array((range(20)))
scores1 = np.array((range(50)),'float32')-25
scores2 = np.ones_like(scores1)

scores = np.array(([scores1,scores2]))
softmax = tf.keras.activations.sigmoid(tf.convert_to_tensor(np.transpose(scores)))
softmax=softmax.numpy()

print(softmax)
print(scores[0])
plt.plot(softmax)