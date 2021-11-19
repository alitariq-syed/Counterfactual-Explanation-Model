# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:10:12 2020

@author: Ali
"""
import numpy as np
import matplotlib.pyplot as plt

file='./create_training_data/0/10_delta_matrix_class_specific.npy'

delta_matrix = np.load(file,allow_pickle=True)

padded_matrix = np.ones((4,128))*-100
padded_matrix[0,:] = delta_matrix[0]
padded_matrix[1,:] = delta_matrix[1]
padded_matrix[2,0:64] = delta_matrix[2]
padded_matrix[3,0:32] = delta_matrix[3]


binarized_matrix = np.zeros((4,128))
binarized_matrix[0,:] = delta_matrix[0]>0
binarized_matrix[1,:] = delta_matrix[1]>0
binarized_matrix[2,0:64] = delta_matrix[2]>0
binarized_matrix[3,0:32] = delta_matrix[3]>0

plt.plot(delta_matrix[0]), plt.title('delta_matrix - layer 0 - GT'),plt.show()

print(np.std(delta_matrix[0]))