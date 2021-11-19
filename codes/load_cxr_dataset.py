# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:40:49 2020

@author: Ali
"""
import pandas as pd
from glob import glob
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#%%
def create_cxr_dataframes():
    database_path_local = 'D:/Datasets/ChestXray-14/'
    database_path_kaggle = '../input/data/'
    data_path = database_path_local
    
    all_xray_df = pd.read_csv(filepath_or_buffer=data_path+'/Data_Entry_2017.csv')
    all_image_paths = {os.path.basename(x): x for x in 
                       glob(os.path.join(data_path, 'images*', '*', '*.png'))}
    print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
    all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
    
    print(all_xray_df.loc[1])
    print(all_xray_df.shape[0])
    #%%
    #all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
    from itertools import chain
    all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x)>0]
    print('All Labels ({}): {}'.format(len(all_labels), all_labels))
    for c_label in all_labels:
        if len(c_label)>1: # leave out empty labels
            all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
    all_xray_df.sample(3)
    # keep at least 1000 cases
    MIN_CASES = 0 #1000
    all_labels = [c_label for c_label in all_labels if all_xray_df[c_label].sum()>MIN_CASES]
    print('Clean Labels ({})'.format(len(all_labels)), 
          [(c_label,int(all_xray_df[c_label].sum())) for c_label in all_labels])
    label_map = all_labels
    #%%
    from sklearn.model_selection import train_test_split
    official_split = True
    
    if official_split:
        print('Using official split')
        official_train_split = pd.read_csv(filepath_or_buffer=data_path+'/train_val_list.txt',header=None)
        official_test_split  = pd.read_csv(filepath_or_buffer=data_path+ '/test_list.txt',header=None)
        
        train_df=all_xray_df.loc[all_xray_df['Image Index'].isin(official_train_split[0])]
        train_df, valid_df = train_test_split(train_df, ##splitting train into train+validation set
                                           test_size = 0.13, #0.25
                                           random_state = 2018,
                                           stratify = train_df['Finding Labels'].map(lambda x: x[:4]))
    
        test_df=all_xray_df.loc[all_xray_df['Image Index'].isin(official_test_split[0])]
    
    
    else:
        print('Using random split')
        train_df, test_df = train_test_split(all_xray_df, 
                                           test_size = 0.125, #0.25 #0.009 == 1000 images
                                           random_state = 2018,
                                           stratify = all_xray_df['Finding Labels'].map(lambda x: x[:4]))
        #use 12.5% of test split as the whole database for reducing training time
        train_df, test_df = train_test_split(test_df, 
                                                   test_size = 0.25, #0.25 #0.009 == 1000 images
                                                   random_state = 2018,
                                                   stratify = test_df['Finding Labels'].map(lambda x: x[:4]))
        valid_df = test_df
        
    print('train', train_df.shape[0],'valid', valid_df.shape[0], 'test', test_df.shape[0])
    return label_map, train_df, test_df, valid_df

#%%
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    df_gen = img_data_gen.flow_from_dataframe(in_df,
                                              x_col =path_col,
                                              y_col = y_col,
                                              class_mode = 'raw',#other multi_output  sparse
                                              **dflow_args)
    return df_gen

#%%
def load_cxr_dataset(train_df, test_df, valid_df, all_labels, batch_size,preprocess_input,augment):
    #augment = 0
    imagenet_weights = 1
    
    IMG_SIZE = (224, 224)# B0
    # IMG_SIZE = (380, 380)# B4
    #batch_size = 64
    
    if imagenet_weights:
        print('using imagenet pretrained weights')
        img_shape = (224,224,3)
        color_mode = 'rgb'
        test_idg  = ImageDataGenerator(preprocessing_function = preprocess_input)

    else:
        print('training on CXR dataset from scratch')
        img_shape = (224,224,3)
        color_mode = 'rgb'#'grayscale'
        test_idg  = ImageDataGenerator(rescale=1./255)

    
    #test_idg  = ImageDataGenerator(rescale=1./255)
    valid_idg = test_idg
    if augment:
        print('with augmentation')
        train_idg = ImageDataGenerator(
                                  horizontal_flip = False,
                                  #rescale=1./255,
                                  vertical_flip = False, 
                                  height_shift_range= 0.2, 
                                  width_shift_range=0.2, 
                                  rotation_range=15, 
                                  shear_range = 0.2,
                                  fill_mode = 'nearest',#''nearest
                                  zoom_range=0.2,
                                  preprocessing_function = preprocess_input
                                  )
    else:
        print('without augmentation')
        train_idg = test_idg
    ##%%
    print('loading training dataframes...')
    train_gen = flow_from_dataframe(train_idg, train_df, 
                             path_col = 'path',
                                y_col = all_labels ,
                          target_size = IMG_SIZE,
                           color_mode = color_mode,
                           batch_size = batch_size,
                              shuffle = True)
    
    test_gen = flow_from_dataframe(test_idg, test_df, 
                                 path_col = 'path',
                                y_col = all_labels,
                                target_size = IMG_SIZE,
                                 color_mode = color_mode,
                                batch_size = batch_size,
                                shuffle=0) 
    
    valid_gen = flow_from_dataframe(valid_idg, valid_df, 
                                 path_col = 'path',
                                y_col = all_labels,
                                target_size = IMG_SIZE,
                                 color_mode = color_mode,
                                batch_size = batch_size,
                                shuffle=0)
    return train_gen, test_gen, valid_gen