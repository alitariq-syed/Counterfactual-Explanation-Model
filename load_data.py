# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 23:30:06 2022

@author: Ali
"""
#%%

from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys

#%%
from config import args, KAGGLE
from load_base_model import preprocess_input
from codes.load_cxr_dataset import create_cxr_dataframes, load_cxr_dataset

#%%
if args.dataset =='mnist':
    batch_size = 32
else:
    batch_size =16# 16 for analysis, 32 for training

top_activation = 'softmax'
loss = categorical_crossentropy    
print("batch_size: ",batch_size)

if args.dataset == 'mnist':
    num_classes=10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train,-1)
    x_test = np.expand_dims(x_test,-1)
    input_shape = (batch_size,28,28,1)
    label_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

elif args.dataset == 'cifar10':
    num_classes=10
    print('cifar-10 dataset')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_shape = (batch_size,32,32,3)
    label_map = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

elif args.dataset == 'CUB200':                
    print('CUB200-2011 dataset')
    input_shape = (batch_size,224,224,3)
    
    official_split=True
    lab_system=False
    if KAGGLE:
        if official_split:
            print('using official split on KAGGLE')
            base_path = '/kaggle/input/cub-train-test-official-split/train_test_split'
            #sys.exit()
        else:
            base_path = '/kaggle/temp/CUB_200_2011'
    elif lab_system:
        base_path ='D:/Ali Tariq/CUB_200_2011'
    else:
        base_path = 'G:/CUB_200_2011/CUB_200_2011'
    
    if official_split:
        data_dir =base_path+'/train_test_split/train/'
        data_dir_test =base_path+'/train_test_split/test/'
        if KAGGLE:
            label_map = np.loadtxt(fname='/kaggle/input/cub-train-test-official-split/classes.txt',dtype='str')
        else:
            label_map = np.loadtxt(fname=base_path + '/classes.txt',dtype='str')
        label_map = label_map[:,1].tolist()
        print('using official split')
    else:
        data_dir =base_path+'/images/'
        label_map = np.loadtxt(fname=base_path + '/classes.txt',dtype='str')
        label_map = label_map[:,1].tolist()
    
    #if train/test on small subset
    #label_map = label_map[8:12]# 4 common bird classes
    
    num_classes=len(label_map)
    print("num classes:", num_classes)

elif args.dataset == 'BraTS':                
    print('BraTS 2020 dataset')
    input_shape = (batch_size,240,240,3)
    
    lab_system=False

    if lab_system:
        base_path ='D:/Datasets/Brats_2020_Original'
    else:
        base_path = 'G:/BraTS_2020'
    
    data_dir =base_path+'/classification_data/train/'
    data_dir_test =base_path+'/classification_data/test/'
    label_map = ['non_tumor', 'tumor']

    num_classes=len(label_map)
    print("num classes:", num_classes)

elif args.dataset == 'NIST':                
    print('NIST-SD04 multilabelled remove dataset')
    input_shape = (batch_size,224,224,3)
    
    lab_system=False

    if lab_system:
        base_path ='D:/Datasets/Brats_2020_Original'
    else:
        base_path = 'G:/NIST_multlabel_removed_Test_Train'
    
    data_dir =base_path+'/Train/'
    data_dir_test =base_path+'/Test/'
    label_map = ['Arch', 'Left loop', 'Right loop', 'Tented arch', 'Whorl']

    num_classes=len(label_map)
    print("num classes:", num_classes)
    
elif args.dataset == 'cxr1000':
    print('CXR-1000 dataset')
    num_classes=15
    input_shape = (batch_size,224,224,3)
    label_map, train_df, test_df, valid_df = create_cxr_dataframes()
    all_labels = label_map
    top_activation = 'sigmoid'
    loss = binary_crossentropy    

elif args.dataset == 'catsvsdogs':
    print('catsvsdogs dataset')
    num_classes=2
    input_shape = (batch_size,224,224,3)
    lab_system = False
    if lab_system:
        data_dir ='D:/Ali Tariq/catsvsdogs/train/'
        data_dir_test ='D:/Ali Tariq/catsvsdogs/test/'    
    else:
        data_dir ='G:/catsvsdogs/train/'
        data_dir_test ='G:/catsvsdogs/test/'

    label_map = ['cat',  'dog']

elif args.dataset == 'VOC2010':
    print('VOC2010-animals dataset')
    num_classes=6
    input_shape = (batch_size,224,224,3)
    lab_system = False
    if lab_system:
        data_dir ='D:/Ali Tariq/VOCdevkit/VOC_animals/'
        #data_dir_test ='D:/Ali Tariq/catsvsdogs/test/'    
    else:
        #data_dir ='G:/VOCdevkit/VOC_animals_one/'
        data_dir ='G:/VOCdevkit/VOC_animals/'
        #data_dir_test ='G:/catsvsdogs/test/'

    label_map = ['bird',  'cat', 'cow', 'dog', 'horse', 'sheep']#['cat']#

else:
    print('unknown dataset')
    sys.exit()

#%% resize cifar10 for VGG minimum required image size
# if (args.model=='VGG16/' and args.dataset == 'cifar10'):
#     # resize train set
#     X_train_resized = []
#     for img in x_train:
#       X_train_resized.append(np.resize(img, (48,48,3)))
      
#     X_train_resized = np.array(X_train_resized)
#     print(X_train_resized.shape)# resize test set
#     X_test_resized = []
#     for img in x_test:
#       X_test_resized.append(np.resize(img, (48,48,3)))
      
#     X_test_resized = np.array(X_test_resized)
#     print(X_test_resized.shape)


#%%
if args.dataset != 'mnist':
    if args.imagenet_weights:
        print('using imagenet_weights')
        rescale = 1./1
    else:
        print('not using imagenet_weights')
        preprocess_input = None
        rescale = 1./255

    if args.dataset == 'cifar10':
        imgDataGen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                        rescale = rescale)

        train_gen = imgDataGen.flow(x_train, y_train, batch_size = batch_size,shuffle= False)
        test_gen  = imgDataGen.flow(x_test, y_test, batch_size = batch_size,shuffle= False)
    elif args.dataset == 'CUB200' or args.dataset == 'BraTS'  or args.dataset == 'NIST':
        augment = False
        if not augment:
            imgDataGen = ImageDataGenerator(preprocessing_function = preprocess_input, 
                                            rescale = rescale,
                                            validation_split= 0.1 if not args.find_global_filters else None)
        else:
            imgDataGen = ImageDataGenerator(preprocessing_function = preprocess_input, 
                                        rescale = rescale,
                                            validation_split= 0.1 if not args.find_global_filters else None,
                                        
                              height_shift_range= 0.2, 
                              width_shift_range=0.2, 
                              rotation_range=15, 
                              shear_range = 0.2,
                              fill_mode = 'nearest',#''nearest#reflect
                              zoom_range=0.2)
        
        train_gen = imgDataGen.flow_from_directory(data_dir,
                                target_size=(input_shape[1], input_shape[2]),
                                color_mode='rgb',
                                class_mode='categorical',
                                batch_size=batch_size,
                                shuffle=True if not args.find_global_filters else False,
                                seed=None,
                                subset='training'if not args.find_global_filters else None,
                                interpolation='nearest',
                                #all classes for base model; binary classes for CF model
                                #classes = ['cat', 'dog'] if args.train_counterfactual_net else label_map)#['cat', 'dog'])
                                classes = [label_map[args.alter_class]] if (args.train_counterfactual_net and args.choose_subclass) else label_map)#['cat', 'dog'])
                                #classes = label_map)
        test_gen  = imgDataGen.flow_from_directory(data_dir,
                                target_size=(input_shape[1], input_shape[2]),
                                color_mode='rgb',
                                class_mode='categorical',
                                batch_size=batch_size,
                                shuffle=False,
                                seed=None,
                                subset='validation' if not args.find_global_filters else None,
                                interpolation='nearest',
                                #classes = ['cat', 'dog'] if args.train_counterfactual_net else label_map)#['cat', 'dog'])
                                classes = [label_map[args.alter_class]] if (args.train_counterfactual_net and args.choose_subclass) else label_map)#['cat', 'dog'])
                               # classes = label_map)
        
        # for visualization, dont use preprocessed image
        if args.imagenet_weights:
            imgDataGen_nopreprocess = ImageDataGenerator(#preprocessing_function = preprocess_input, 
                                            rescale = 1./255,
                                            validation_split=0.1)
            
            train_gen_nopreprocess = imgDataGen_nopreprocess.flow_from_directory(data_dir,
                                    target_size=(input_shape[1], input_shape[2]),
                                    color_mode='rgb',
                                    class_mode='categorical',
                                    batch_size=batch_size,
                                    shuffle= True if not args.find_global_filters else False,
                                    seed=None,
                                    subset='training',
                                    interpolation='nearest',
                                    #classes = ['cat', 'dog'] if args.train_counterfactual_net else label_map)#['cat', 'dog'])
                                    classes = [label_map[args.alter_class]] if (args.train_counterfactual_net and args.choose_subclass) else label_map)#['cat', 'dog'])
                                    #classes = label_map)
            test_gen_nopreprocess  = imgDataGen_nopreprocess.flow_from_directory(data_dir,
                                    target_size=(input_shape[1], input_shape[2]),
                                    color_mode='rgb',
                                    class_mode='categorical',
                                    batch_size=batch_size,
                                    shuffle=False,
                                    seed=None,
                                    subset='validation',
                                    interpolation='nearest',
                                    #classes = ['cat', 'dog'] if args.train_counterfactual_net else label_map)#['cat', 'dog'])
                                    classes = [label_map[args.alter_class]] if (args.train_counterfactual_net and args.choose_subclass) else label_map)#['cat', 'dog'])
                                    #classes = label_
        if (args.dataset == 'CUB200' and official_split) or args.dataset=='BraTS' or args.dataset == 'NIST':
            #actual unseen test set
            imgDataGen_official_split = ImageDataGenerator(preprocessing_function = preprocess_input)
            actual_test_gen  = imgDataGen_official_split.flow_from_directory(data_dir_test,
                            target_size=(input_shape[1], input_shape[2]),
                            color_mode='rgb',
                            class_mode='categorical',
                            batch_size=batch_size,
                            shuffle=False,
                            seed=111,
                            #subset='validation',
                            interpolation='nearest',#)
                            classes = [label_map[args.alter_class]] if (args.train_counterfactual_net and args.choose_subclass) else label_map)#['cat', 'dog'])

            imgDataGen_official_split_nopreprocess = ImageDataGenerator(rescale = 1./255)
            actual_test_gen_nopreprocess  = imgDataGen_official_split_nopreprocess.flow_from_directory(data_dir_test,
                            target_size=(input_shape[1], input_shape[2]),
                            color_mode='rgb',
                            class_mode='categorical',
                            batch_size=batch_size,
                            shuffle=False,
                            seed=111,
                            #subset='validation',
                            interpolation='nearest',
                            classes = [label_map[args.alter_class]] if (args.train_counterfactual_net and args.choose_subclass) else label_map)#['cat', 'dog'])

    elif args.dataset == 'cxr1000':
        train_gen, test_gen, valid_gen = load_cxr_dataset(train_df, test_df, valid_df, all_labels, batch_size,preprocess_input,augment=0)
        
else:
    print('not using imagenet_weights')
    
    if args.dataset == 'NIST':
        pass
    else:
        imgDataGen = ImageDataGenerator(rescale = 1./255)
        
        train_gen = imgDataGen.flow(x_train, y_train, batch_size = batch_size,shuffle= True)
        test_gen  = imgDataGen.flow(x_test, y_test, batch_size = batch_size,shuffle= False)
    
