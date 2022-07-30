# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 02:04:44 2022

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
#%%
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Softmax, GlobalAveragePooling2D
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import math
from tf_explain_modified.core.grad_cam import GradCAM
import time
from sklearn.preprocessing import binarize
#%%
from codes.support_functions import restore_original_image_from_array
from codes.find_agreement_global_MC import find_agreement_global_MC

#%%
from config import args, weights_path, KAGGLE, pretrained_weights_path
from load_data import top_activation, num_classes, train_gen, label_map, actual_test_gen
from load_base_model import base_model
from load_CFE_model import load_cfe_model
#%%
np.random.seed(seed=100)


    
assert(args.find_global_filters==False) #to make sure data generators are setup properly
assert(args.counterfactual_PP)
assert(args.train_all_classes)

#%% create CFE model


    
#%%

if args.dataset == 'CUB200' or args.dataset == 'BraTS' or args.dataset == 'NIST': 
    test_gen =train_gen if args.find_global_filters else actual_test_gen# train_gen#actual_test_gen
    #test_gen_nopreprocess = train_gen_nopreprocess if args.find_global_filters else actual_test_gen_nopreprocess #train_gen_nopreprocess[0]#actual_test_gen_nopreprocess
    # print("using traingen gen data") if args.find_global_filters else print("using testgen gen data")

gen=test_gen
batches=math.ceil(gen.n/gen.batch_size)
    
         
gen.reset() #resets batch index to 0

default_accuracy = False
pred_label = []
gt_label = []
loaded_cfe_model = -1
correct_decisions_made = 0
correct_inferred_made_false = 0
wrong_pred_still_wrong=0
all_class_MC_filters=[]
all_class_MC_probs = []
start = time.time()
for loop in range(num_classes):#range(1):#num_classes):
    args.alter_class = loop#9#loop
    
    #assert(False)#verify if model is loaded with alter_class from above line; or does it require passing the class
    
    combined = load_cfe_model()
    gen.reset()
    print(label_map[loop])
    
    MC_filters=[]
    MC_probs = []
    for k in range(batches):
        sys.stdout.write("\r batch %i of %i" % (k, batches))    
        x_batch_test,y_batch_test = next(gen)
           
        #%%        
        alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(x_batch_test)

        MC_filters.append(modified_mean_fmap_activations)
        MC_probs.append(alter_prediction)
    #%%
    all_class_MC_filters.append(np.vstack(MC_filters))
    all_class_MC_probs.append(np.vstack(MC_probs))

np.save(file=weights_path+"/all_class_MC_filters_class.npy",arr=all_class_MC_filters)    
np.save(file=weights_path+"/all_class_MC_probs_class.npy",arr=all_class_MC_probs)    
end = time.time()


print("time taken: ",end - start)