# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:47:28 2022

@author: Ali
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def find_agreement_global_MC(pred_MC,target_class, args):
    
    save_folder = "./model_debugging_work/"
    mName = args.model[:-1]+'_'+args.dataset
    
    try:
        filter_magnitude_cf = np.load(file= save_folder+mName+"_normalized_filter_magnitude_cf_"+str(target_class)+"_train_set.np.npy")
        filter_histogram_cf = np.load(file= save_folder+mName+"_filter_histogram_cf_"+str(target_class)+"_train_set.np.npy")
    except:
        return 0
    
    
    # plt.plot(filter_histogram_cf), plt.ylim([0, np.max(filter_histogram_cf)+1]),plt.xlabel("Filter number"),plt.ylabel("Filter activation count"), plt.show()
    # plt.plot(filter_magnitude_cf/max(filter_histogram_cf)),plt.xlabel("Filter number"),plt.ylabel("Avg. activation magnitude"), plt.ylim([0, np.max(filter_magnitude_cf/np.max(filter_histogram_cf))+1]), plt.show()


    # plt.plot(pred_MC),plt.ylim([0, np.max(pred_MC)+1]), plt.title('thresh_mean_fmap_activations'), plt.show()


    pred_MC_binary = np.copy(pred_MC) 
    for i in tf.where(pred_MC>0):
        pred_MC_binary[tuple(i)]=1.0
    
    filter_histogram_cf_binary = np.copy(filter_histogram_cf)
    for i in tf.where(filter_histogram_cf>0):
        filter_histogram_cf_binary[tuple(i)]=1.0
                    
                    
    overlap = np.sum(pred_MC_binary * filter_histogram_cf_binary)/np.sum(filter_histogram_cf_binary)
    
    return overlap