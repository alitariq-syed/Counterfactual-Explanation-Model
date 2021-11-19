# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:51:47 2020

@author: Ali
"""
import numpy as np

img_ind =14

save_path = './create_training_data/9'

loaded_matrix = np.load(file=save_path+'/'+str(img_ind)+'_matrix.npy')
loaded_pred_class = np.load(file=save_path+'/'+str(img_ind)+'_pred_class.npy')
loaded_stats = np.load(file=save_path+'/'+str(img_ind)+'_stats.npy')
loaded_delta_matrix = np.load(file=save_path+'/'+str(img_ind)+'_delta_matrix.npy')