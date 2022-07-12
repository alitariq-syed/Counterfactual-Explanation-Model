# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 23:59:52 2022

@author: Ali
"""
#%%
import tensorflow as tf
print('TF version: ', tf.__version__)

#%% 
"""fix for issue: cuDNN failed to initialize"""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('...GPU set_memory_growth successfully set...')

else:
    print('...GPU set_memory_growth not set...')

import numpy as np

#%%
from config import args
from load_data import label_map, actual_test_gen
from load_CFE_model import model
from codes.model_accuracy_with_disabled_filters import model_accuracy_filters

#%%


args.alter_class = 199

#%%

save_folder = "./model_debugging_work/epoch_"+str(args.cfe_epochs)+"/"+label_map[args.alter_class]+"/"

mName = args.model[:-1]+'_'+args.dataset

filter_magnitude_cf = np.load(file= save_folder+mName+"_normalized_filter_magnitude_cf_"+str(args.alter_class)+"_train_set.np.npy")
filter_histogram_cf = np.load(file= save_folder+mName+"_filter_histogram_cf_"+str(args.alter_class)+"_train_set.np.npy")    

freq_thresh = 0.25 #percent
filter_histogram_cf_binary = tf.where(filter_histogram_cf>(freq_thresh*max(filter_histogram_cf)),1.0,0.0)

enabled_filters= 1-filter_histogram_cf_binary
print("\n Global PP filters disabled: ", np.sum(filter_histogram_cf_binary))

#%%
#random test: randomly disable some number of filters and see change in performance
random_filters_alter_class = np.zeros_like(filter_histogram_cf) 
while np.sum(random_filters_alter_class)!=40:
    rndIndx = np.random.randint(512)
    random_filters_alter_class[rndIndx]=1.0          
random_enabled_filters = 1-random_filters_alter_class
# print("randoms filters disabled:", np.sum(random_filters_alter_class))          

#%%
#enabled_filters = np.ones(512)
test_acc, test_loss, c_report = model_accuracy_filters(model,actual_test_gen, enabled_filters, args)
#print(c_report)