# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 01:04:39 2022

@author: Ali
"""
import numpy as np
#%%
from config import args
from load_data import label_map


#%%



save_folder = "./model_debugging_work/epoch_"+str(args.cfe_epochs)+"/"+label_map[args.alter_class]+"/"
mName = args.model[:-1]+'_'+args.dataset

acc_A = np.load("model_accuracies_original.npy")
# acc_B = np.load(file=save_folder+mName+"_global_disabled_accuracies_"+str(args.alter_class)+".npy")
acc_B = np.load("model_accuracies_debugged.npy")

diff = acc_B - acc_A

side_by_side = np.vstack([acc_A , acc_B,  diff])