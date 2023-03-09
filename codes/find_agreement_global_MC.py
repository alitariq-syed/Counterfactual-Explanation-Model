# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:47:28 2022

@author: Ali
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

def find_agreement_global_MC(freq_thresh,pred_MC,target_class, args, class_name, class_weights):
    enable_prints = False
    #save_folder = "./model_debugging_work/"
    save_folder = "./model_debugging_work/epoch_"+str(args.cfe_epochs)+"/"+class_name+"/"
    
    mName = args.model[:-1]+'_'+args.dataset
    
    try:
        filter_magnitude_cf = np.load(file= save_folder+mName+"_normalized_filter_magnitude_cf_"+str(target_class)+"_train_set.np.npy")
        filter_histogram_cf = np.load(file= save_folder+mName+"_filter_histogram_cf_"+str(target_class)+"_train_set.np.npy")
    except:
        raise Exception("Global MC Filters not loaded") 
        #return 0
    
    
    if enable_prints:
        plt.plot(filter_histogram_cf), plt.ylim([0, np.max(filter_histogram_cf)+1]),plt.xlabel("Filter number"),plt.ylabel("Filter activation count"), plt.show()
        plt.plot(filter_magnitude_cf/max(filter_histogram_cf)),plt.xlabel("Filter number"),plt.ylabel("Avg. activation magnitude"), plt.ylim([0, np.max(filter_magnitude_cf/np.max(filter_histogram_cf))+1]), plt.show()


        plt.plot(pred_MC), plt.ylim([0, np.max(pred_MC)+1]),plt.xlabel("Filter number"),plt.ylabel("Filter activation magnitude"), plt.show()


    pred_MC_binary = tf.where(pred_MC>0,1.0,0.0)

    
    freq_thresh = freq_thresh #percent
    filter_histogram_cf_binary = tf.where(filter_histogram_cf>(freq_thresh*max(filter_histogram_cf)),1.0,0.0)
    filter_magnitude_cf_thresholded = (filter_magnitude_cf*filter_histogram_cf_binary)/max(filter_histogram_cf)
                    
#%%                
    recall = np.sum(pred_MC_binary * filter_histogram_cf_binary)/np.sum(filter_histogram_cf_binary)
    precision = np.sum(pred_MC_binary * filter_histogram_cf_binary)/np.sum(pred_MC_binary)
    avg_prec_recall = (precision+precision)/2
    F1 = 2 * (precision * recall) / (precision + recall)
#%%
    a=pred_MC#/max(filter_histogram_cf)
    b=filter_magnitude_cf_thresholded
    with_mag = np.sum((pred_MC_binary * filter_histogram_cf_binary * pred_MC ) *(abs(a-b)))
    
    mag_similarity = np.sum(abs(a-b)) #lower is better; but with -neg sign, higher is better

#%%    
    class_weights_score = np.sum((class_weights*pred_MC_binary)*pred_MC)/np.sum((class_weights*filter_histogram_cf_binary)*filter_histogram_cf)
    
#%%
    overlap = precision
    #overlap=(class_weights_score+mag_similarity+F1)/3
    #%%

    return [recall,precision, F1, mag_similarity, class_weights_score]