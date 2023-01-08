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
from load_data import label_map, train_gen, actual_test_gen
from load_CFE_model import model
from codes.model_accuracy_with_disabled_filters import model_accuracy_filters

#%%

if args.dataset == 'CUB200' or args.dataset == 'BraTS' or args.dataset == 'NIST': 
    test_gen =actual_test_gen #if args.find_global_filters else actual_test_gen# train_gen#actual_test_gen
    #test_gen_nopreprocess = train_gen_nopreprocess if args.find_global_filters else actual_test_gen_nopreprocess #train_gen_nopreprocess[0]#actual_test_gen_nopreprocess
    # print("using traingen gen data") if args.find_global_filters else print("using testgen gen data")

#%%

args.alter_class = 9
print("for class: ", label_map[args.alter_class])
#%%

save_folder = "./model_debugging_work/epoch_"+str(args.cfe_epochs)+"/"+label_map[args.alter_class]+"/"

mName = args.model[:-1]+'_'+args.dataset

filter_magnitude_cf = np.load(file= save_folder+mName+"_normalized_filter_magnitude_cf_"+str(args.alter_class)+"_train_set.np.npy")
filter_histogram_cf = np.load(file= save_folder+mName+"_filter_histogram_cf_"+str(args.alter_class)+"_train_set.np.npy")    

freq_thresh = 0.0 #percent
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
enabled_filters = np.ones(512)
test_acc, test_loss, c_report = model_accuracy_filters(model,test_gen, enabled_filters, args)

#%%

class_recalls = []
for i in range(200):
    class_recalls.append(c_report[str(i)]['recall'])
class_recalls=np.asarray(class_recalls)    
#np.save(file=save_folder+mName+"_global_disabled_accuracies_"+str(args.alter_class),arr=class_recalls)
#print(c_report)
# np.save(file="model_accuracies_original",arr=class_recalls)
# np.save(file="model_accuracies_debugged_7064",arr=class_recalls)
