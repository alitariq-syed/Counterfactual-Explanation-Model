# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:43:27 2020

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
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Softmax, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import math
from tf_explain_modified.core.grad_cam import GradCAM
import datetime
from tqdm import tqdm # to monitor progress
import argparse
import os, sys

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers


from models10 import MySubClassModel
from codes.compute_filter_importance import save_filter_importance, test_filter_importance,test_filter_importance_in_code_method, plot_filter_importance,check_top_filter_importance,save_filter_importance_batch, check_histogram_top_filter_result
from codes.load_cxr_dataset import create_cxr_dataframes, load_cxr_dataset
from codes.support_functions import print_filter_classes_1, print_filter_classes_2, save_interpretable_parameters
from codes.find_filter_class import find_filter_class
from codes.train_counterfactual_net import train_counterfactual_net
from codes.support_functions import get_heatmap_only, restore_original_image_from_array
from codes.filter_visualization_top_k import filter_visualization_top_k
from codes.filter_visualization_same_image import filter_visualization_same_image
from codes.model_accuracy_with_disabled_filters import model_accuracy_filters
#%%
KAGGLE = False
parser = argparse.ArgumentParser(description='Interpretable CNN')
parser.add_argument('--interpretable',default = False) # keep false for CFE work
parser.add_argument('--full_standard',default = True) # keep True for CFE work... dont add extra cnn layer to be comparable with interpretable model. Make completely standalone model.... 

#choose wheter to train a CF model for a given base model or train a base model from scratch
parser.add_argument('--create_counterfactual_combined' ,default = True)## create CF model for a pretrained base model or train a new base model
parser.add_argument('--filter_visualization' ,default = True) # find top k highest and lowest activation magnitudes for the target filter and the corresponding images

parser.add_argument('--user_evaluation' ,default = True) # save images

# CF model args
parser.add_argument('--train_counterfactual_net' ,default = False)## 
parser.add_argument('--choose_subclass' ,default = False, type=np.bool)## choose subclass for training on
parser.add_argument('--counterfactual_PP' ,default = True)## whether to generate filters for PP  or PN case 
parser.add_argument('--resume_counterfactual_net' ,default = False)## False = train CF model from scratch; True = resume training CF model
parser.add_argument('--test_counterfactual_net' ,default = False)## 
parser.add_argument('--load_counterfactual_net',default = True)
parser.add_argument('--resume', default =True) # load saved weights for base model
parser.add_argument('--alter_class', default = 9, type = np.int32) # alter class #misclassified classes 9-170
parser.add_argument('--analysis_class', default = 9, type = np.int32) # class for which images are loaded and analyzed
parser.add_argument('--find_global_filters', default = False) # perform statistical analysis to find the activation magnitude of all filters for the alter class and train images of alter class
parser.add_argument('--alter_class_2', default = 0, type = np.int32) # alter class for 2nd example, 9, 170, 25, 125, 108
parser.add_argument('--cfe_epochs', default = 30, type = np.int32 ) #100 for mnist, 200 for CUB
parser.add_argument('--l1_weight', default = 2, type = np.float32) # 2 default
parser.add_argument('--save_logFile', default = False,type=np.bool) #

#parser.add_argument('--pretrained', default = False) # load self-pretrained model for cifar dataset... i.e. load base model already trained on cifar-10


#base model parameters
parser.add_argument('--dataset',default = 'CUB200')#NIST, BraTS,mnist, cifar10, CUB200, #cxr1000, #catsvsdogs, #VOC2010
parser.add_argument('--save_directory',default = './trained_weights/')
parser.add_argument('--train_using_builtin_fit_method',default = True)#for training base model easily
parser.add_argument('--train',default = False)
parser.add_argument('--fine_tune',default = False) # fine tune all weights after transfer learning step (CUB dataset)
parser.add_argument('--test', default = False)
parser.add_argument('--model',default = 'VGG16/')#myCNN, VGG16, resnet50,efficientnet, inceptionv3
parser.add_argument('--imagenet_weights',default = True) #use imageNet pretrained weights (True for CUB dataset)

#interpretable model args
parser.add_argument('--find_filter_class', default = False) # load retrained model and assign class to each filter by check mean activation per filter per class
parser.add_argument('--filter_modified_directly', default = True)
parser.add_argument('--loss_compute', default = True)#False = forward only
parser.add_argument('--high_capacity_model', default = True)#
parser.add_argument('--fixed_classes', default = True)#idea 2: fine tune from forward only with fixed classes
parser.add_argument('--fixed_classes_reduce_loss', default = True)#False = forward only masked with fixed filter class. issue: 100% training accuracy but 10% testing acc
parser.add_argument('--test_filter_importance', default = False)#for testing idea 2
parser.add_argument('--save_filter_importance', default = False)#for testing idea 2
parser.add_argument('--analyze_filter_importance', default = False)#for testing idea 2
parser.add_argument('--save_filter_fmap', default = False)#save filter fmap as well
parser.add_argument('--save_top_layer', default = True)#save top layer filter data only
parser.add_argument('--visualize_fmaps', default = False)
parser.add_argument('--filter_category_method',default = 'own_reduce_loss')   # paper --> similar to paper implementation---assign filter with categories during training by accumulating batch-wise max activations
                                                                    # own_reduce_loss --> our idea - pre-assign filter categories during forward pass over all the data, based on pretrained weights and feature maps
np.random.seed(seed=100)
 
if KAGGLE: args = parser.parse_known_args()[0] 
else: args = parser.parse_args()

if args.train_counterfactual_net:
    assert(args.find_global_filters==False)
    #make sure training generators are setup properly
if args.interpretable:
    if args.filter_category_method=='paper':
        print('filter category assignment --> paper method')
    else:
        print('filter category assignment --> our idea')
    weights_path = args.save_directory+args.model+args.dataset+'/interpretable/filter_category_method_'+str(args.filter_category_method)
    log_path  = './logs/'+args.model+args.dataset+'/interpretable/filter_category_method_'+str(args.filter_category_method)
    filter_data_path = './create_training_data/'+args.model+args.dataset+'/interpretable/filter_category_method_'+str(args.filter_category_method)
else:
    weights_path = args.save_directory+args.model+args.dataset+'/standard'
    log_path  = './logs/'+args.model+args.dataset+'/standard'
    filter_data_path = './create_training_data/'+args.model+args.dataset+'/standard' #directory for saving filter importance training data


if not os.path.exists(weights_path):
    os.makedirs(weights_path)    
logging = args.save_logFile # save file not required if code executed in jupyter notebook
if logging and args.train_counterfactual_net: 
    if args.counterfactual_PP:
        sys.stdout = open(file=weights_path+"/console_output_"+str(args.alter_class)+".txt", mode="w")    
    else:
        sys.stdout = open(file=weights_path+"/console_output_PN_"+str(args.alter_class)+".txt", mode="w")    
else: print("not saving log file")
    
print('save_path: ',weights_path)

parser.add_argument('--save_path',default = weights_path)
if KAGGLE: args = parser.parse_known_args()[0] 
else: args = parser.parse_args()

if args.resume:
    print("resuming training")
#print(args)

if args.model == 'VGG16/' or args.model =='myCNN/':
    from tensorflow.keras.applications.vgg16 import VGG16,decode_predictions, preprocess_input
elif args.model == 'resnet50/':
    from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
elif args.model == 'efficientnet/':
    from tensorflow.keras.applications.efficientnet import EfficientNetB0, decode_predictions, preprocess_input
elif args.model == 'inceptionv3/':
    from tensorflow.keras.applications.inception_v3 import InceptionV3, decode_predictions, preprocess_input
else:
    print("incorrect model specified")
    sys.exit()    
print('\n',args, '\n')
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
            print('official split not established for training on KAGGLE')
            sys.exit()
        else:
            base_path = '/kaggle/temp/CUB_200_2011'
    if lab_system:
        base_path ='D:/Ali Tariq/CUB_200_2011'
    else:
        base_path = 'G:/CUB_200_2011/CUB_200_2011'
    
    if official_split:
        data_dir =base_path+'/train_test_split/train/'
        data_dir_test =base_path+'/train_test_split/test/'

        data_dir_user_evaluation ='E:/Medical Imaging Diagnostic (MID) Lab/XAI in MID/Paper submission/Version_2_Elsevier_KBS_format/Revision 1/user evaluation/images for comparison/'

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
                                    shuffle=True,
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

            test_gen_user_eval  = imgDataGen_official_split.flow_from_directory(data_dir_user_evaluation,
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
            test_gen_user_eval_nopreprocess  = imgDataGen_official_split_nopreprocess.flow_from_directory(data_dir_user_evaluation,
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
    
#%%
visualize_cxr_images=0
if visualize_cxr_images:
    t_x, t_y = next(train_gen)
    fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
    for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
        c_ax.imshow(c_x[:,:,0], cmap = 'bone')#, vmin = -1.5, vmax = 1.5)
        c_ax.set_title(', '.join([n_class for n_class, n_score in zip(all_labels, c_y) 
                                 if n_score>0.5]))
        c_ax.axis('off')

#%%
if args.high_capacity_model:
    def MyFunctionalModel():
        inputs = tf.keras.Input(shape = train_gen.x[0].shape)
      
        x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPool2D((2,2))(x)
        x = Conv2D(128, (3, 3), activation='relu',strides=(1,1), name='target_conv')(x)
    
        return tf.keras.Model(inputs, x)
else:
    def MyFunctionalModel():
        inputs = tf.keras.Input(shape = train_gen.x[0].shape)
      
        x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPool2D((2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu',strides=(1,1), name='target_conv')(x)
    
        return tf.keras.Model(inputs, x)

#%%
if args.imagenet_weights:
    print('loading VGG model')
    if args.dataset == 'cxr1000' and False:
        tr = 1
        if tr:
            print('using imagenet weights for CXR dataset')
            vgg = VGG16(weights='imagenet',include_top = True)#top needed to get output dimensions at each layer
                # EfficientNetB0(include_top=True,
                #                weights=None,
                #                input_shape=img_shape,
                #                classes=len(all_labels),
                #                classifier_activation='sigmoid')
            base_model = tf.keras.Model(vgg.input,vgg.layers[-6].output)
            #model.compile(optimizer = optimizers.RMSprop(), loss = 'binary_crossentropy',#adam #weighted_binary_crossentropy #lr=0.001/2#binary_crossentropy
                                       #metrics = ['binary_accuracy'])#,tf.keras.metrics.AUC()])
        else:
            print('loading saved model - NOT IMPLEMENTED YET')
            #model = load_model('../input/efnb0-saved-weights/xray_class_EfficientNetB4_15_class_CEL_heatmap_imagenet_pretrained_weights.05-0.1807.hdf5')
        
        #model.summary()
    elif args.dataset == 'cifar10':
        print('using imagenet weights for cifar10 dataset')
        vgg = VGG16(weights='imagenet',include_top = False,input_shape=(32,32,3))#top needed to get output dimensions at each layer
        freeze=True
        if freeze:
            for layer in vgg.layers:
                layer.trainable = False
        base_model = tf.keras.Model(vgg.input,vgg.layers[-2].output)
    elif args.dataset == 'CUB200' or args.dataset == 'cxr1000' or args.dataset == 'NIST':
        if args.model == 'VGG16/':
           print('using VGG16 imagenet weights for CUB200 dataset')
           vgg = VGG16(weights='imagenet',include_top = False,input_shape=(224,224,3))#top needed to get output dimensions at each layer
           freeze=True
           if freeze:
               for layer in vgg.layers:
                   print (layer.name)
                   if layer.name == '----block5_conv3': continue
                   else: layer.trainable = False
                   
               neuron_sum=0    
               for layer in vgg.layers:
                    # print (layer.name, layer.output.shape)
                    if 'conv' in layer.name: 
                        print (layer.name,'\t\t', layer.output.shape)
                        neuron_sum+=layer.output.shape[-1]
               print("total neurons: ", neuron_sum)                   
           base_model = tf.keras.Model(vgg.input,vgg.layers[-2].output)
        elif args.model == 'resnet50/':
           print('using resnet50 imagenet weights for CUB200 dataset')
           vgg = ResNet50(weights='imagenet',include_top = False,input_shape=(224,224,3))#top needed to get output dimensions at each layer
           freeze=True
           if freeze:
               for layer in vgg.layers:
                   print (layer.name)
                   if layer.name == '----block5_conv3': continue
                   else: layer.trainable = False
           base_model = tf.keras.Model(vgg.input,vgg.output)
        elif args.model == 'efficientnet/':
               print('using efficientnet imagenet weights for CUB200 dataset')
               vgg = EfficientNetB0(weights='imagenet',include_top = False,input_shape=(224,224,3))#top needed to get output dimensions at each layer
               freeze=True
               if freeze:
                   for layer in vgg.layers:
                       print (layer.name)
                       if layer.name == '----block5_conv3': continue
                       else: layer.trainable = False
               base_model = tf.keras.Model(vgg.input,vgg.output)
        elif args.model == 'inceptionv3/':
               print('using inceptionv3 imagenet weights for CUB200 dataset')
               inception_v3 = InceptionV3(weights='imagenet',include_top = False,input_shape=(299,299,3))#top needed to get output dimensions at each layer
               freeze=True
               neuron_sum = 0
               if freeze:
                   for layer in inception_v3.layers:
                       # print (layer.name, layer.output.shape)
                       if 'conv' in layer.name: 
                           print (layer.name,'\t\t', layer.output.shape)
                           neuron_sum+=layer.output.shape[-1]
               print("total neurons: ", neuron_sum)
               base_model = tf.keras.Model(inception_v3.input,inception_v3.output)
    elif args.dataset == 'BraTS':
        if args.model == 'VGG16/':
           print('using VGG16 imagenet weights for BraTS dataset')
           vgg = VGG16(weights='imagenet',include_top = False,input_shape=(240,240,3))#top needed to get output dimensions at each layer
           freeze=True
           if freeze:
               for layer in vgg.layers:
                   print (layer.name)
                   if layer.name == '----block5_conv3': continue
                   else: layer.trainable = False
                   
               neuron_sum=0    
               for layer in vgg.layers:
                    # print (layer.name, layer.output.shape)
                    if 'conv' in layer.name: 
                        print (layer.name,'\t\t', layer.output.shape)
                        neuron_sum+=layer.output.shape[-1]
               print("total neurons: ", neuron_sum)                   
           base_model = tf.keras.Model(vgg.input,vgg.layers[-2].output)
    elif args.dataset == 'catsvsdogs':
        print('using imagenet weights for catsvsdogs dataset')
        vgg = VGG16(weights='imagenet',include_top = False,input_shape=(224,224,3))#top needed to get output dimensions at each layer
        freeze=True
        if freeze:
            for layer in vgg.layers:
                layer.trainable = False
        base_model = tf.keras.Model(vgg.input,vgg.layers[-2].output)
    elif args.dataset == 'VOC2010':
        print('using imagenet weights for VOC2010-animals dataset')
        vgg = VGG16(weights='imagenet',include_top = False,input_shape=(224,224,3))#top needed to get output dimensions at each layer
        freeze=True
        if freeze:
            for layer in vgg.layers:
                layer.trainable = False
        base_model = tf.keras.Model(vgg.input,vgg.layers[-2].output)
else:
    base_model = VGG16(weights=None,include_top = False)
    #base_model = MyFunctionalModel()


#%% create base model
if args.full_standard:
    top_filters = base_model.output_shape[3] # flters in top conv layer (512 for VGG)
    fmatrix = tf.keras.layers.Input(shape=(top_filters))
    #flag = tf.keras.layers.Input(shape=(1))
    
    if args.model == 'VGG16/' or args.model == 'myCNN/':
        x =  MaxPool2D()(base_model.output)
    elif args.model == 'resnet50/':
        x =  base_model.output
    elif args.model == 'efficientnet/':
        x =  base_model.output
    mean_fmap = GlobalAveragePooling2D()(x)
    

    
    #modify base model (once it has been pre-trained separately) to be used with CF model later
    if args.create_counterfactual_combined:
        if args.counterfactual_PP:
            modified_fmap = mean_fmap*fmatrix
        else:#PN
            modified_fmap = mean_fmap+fmatrix
        pre_softmax = Dense(num_classes,activation=None)(modified_fmap)
        out = tf.keras.layers.Activation(top_activation)(pre_softmax)
        model = tf.keras.Model(inputs=[base_model.input, fmatrix], outputs= [out,base_model.output, mean_fmap, modified_fmap,pre_softmax],name='base_model')
        
        if args.counterfactual_PP:
            default_fmatrix = tf.ones((train_gen.batch_size,base_model.output.shape[3]))
        else:
            default_fmatrix = tf.zeros((train_gen.batch_size,base_model.output.shape[3]))
    else:
        if args.model == 'myCNN/':
            dropout_rate = 0.0
            x = mean_fmap#dropout creating issue in training CFE model on mnist dataset for self-created functional CNN model
        else:
            dropout_rate = 0.5
            x = tf.keras.layers.Dropout(dropout_rate)(mean_fmap)
        print("dropout_rate: ", dropout_rate)
        


        x = Dense(num_classes,activation=top_activation)(x)
        if args.train_using_builtin_fit_method:
            model = tf.keras.Model(inputs=base_model.input, outputs= [x])#, base_model.output])
        else:
            model = tf.keras.Model(inputs=base_model.input, outputs= [x, base_model.output])
else:
    model = MySubClassModel(num_classes=num_classes, base_model=base_model, args=args)
    #model = base_model
    model(tf.zeros(input_shape))
    #model.build(input_shape = input_shape)

model.summary()

#model.load_weights('./trained_weights/VGG16/CUB200/standard/model.09-2.3280.hdf5')
#load saved weights
if args.resume:
    if args.model =='myCNN/':
        model.load_weights(filepath=weights_path+'/model_transfer_epoch_50.hdf5')
    else:
        model.load_weights(filepath=weights_path+'/model_fine_tune_epoch_150.hdf5')
    #model.load_weights(filepath=weights_path+'/model.hdf5')

    print("weights loaded")
#if args.pretrained:
#    model.load_weights('./trained_weights/myCNN/cifar10/standard/model.hdf5')
#    print("pretrained weights loaded")


#%% filter visulaization - find topk images that most positvely or negatively activate target filter
T = 100# target filter
k = 5# topK images

if args.filter_visualization and False:
    if args.dataset == 'CUB200': 
        test_gen = actual_test_gen# train_gen#actual_test_gen
    filter_visualization_top_k(model,test_gen,T,k,args,show_images=True)

    sys.exit()
#%% Trains for 5 epochs.
if args.train and args.train_using_builtin_fit_method:
    #if args.model =='myCNN/':
    #    optimizer = optimizers.RMSprop(lr=0.01/10)
    #else:
    optimizer = optimizers.SGD(lr=0.01/10, momentum = 0.9)

    model.compile(optimizer=optimizer, 
                  loss=[loss], 
                  metrics=['accuracy'])


    #%%
    save_path=weights_path+'/model_transfer.{epoch:02d}-{val_loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only = True)
    callbacks_list = [checkpoint]
    #%%
    if args.dataset == 'cxr1000':
        history = model.fit(train_gen, epochs=20, verbose=1, callbacks=callbacks_list, validation_data=valid_gen, shuffle=True)
    else:
        history = model.fit(train_gen, epochs=20, verbose=1, callbacks=callbacks_list, validation_data=test_gen, shuffle=True)
    
    model.save_weights(filepath=weights_path+"/model_transfer_epoch_50.hdf5")
    
    plt.style.use('seaborn')
    #plt.style.available
    #['fivethirtyeight',
     #'seaborn-pastel',
     #'seaborn-whitegrid',
     #'ggplot',
     #'grayscale']
    
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    #plt.savefig(fname='model_accuracy_'+db+'.png')
    
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    #plt.savefig(fname='model_loss_'+db+'.png')
    
    #%% Now fine tune all layers at small learning rate with data augmentation
if args.fine_tune: #args.imagenet_weights:
    augment = True
    if args.dataset == 'cxr1000':
        if augment:
            train_gen, test_gen, valid_gen = load_cxr_dataset(train_df, test_df, valid_df, all_labels, batch_size,preprocess_input,augment=1)
            actual_test_gen = test_gen
    else:            
        if not augment:
            imgDataGen = ImageDataGenerator(preprocessing_function = preprocess_input, 
                                            #rescale = 1./255,
                                            validation_split=0.1)
        else:
            imgDataGen = ImageDataGenerator(preprocessing_function = preprocess_input, 
                                        #rescale = 1./255,
                                        validation_split=0.1,
                                        
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
                                shuffle=True,
                                seed=None,
                                subset='training',
                                interpolation='nearest')#,
                                #all classes for base model; binary classes for CF model
                                #classes = ['cat', 'dog'] if args.train_counterfactual_net else label_map)#['cat', 'dog'])
                                #classes = ['cat'] if args.train_counterfactual_net else label_map)#['cat', 'dog'])
                                #classes = label_map)
        test_gen  = imgDataGen.flow_from_directory(data_dir,
                                target_size=(input_shape[1], input_shape[2]),
                                color_mode='rgb',
                                class_mode='categorical',
                                batch_size=batch_size,
                                shuffle=False,
                                seed=None,
                                subset='validation',
                                interpolation='nearest')#,
                                #classes = ['cat', 'dog'] if args.train_counterfactual_net else label_map)#['cat', 'dog'])
                                #classes = ['cat'] if args.train_counterfactual_net else label_map)#['cat', 'dog'])
                               # classes = label_map)
        
        # for visualization, dont use preprocessed image
        imgDataGen_nopreprocess = ImageDataGenerator(#preprocessing_function = preprocess_input, 
                                        rescale = 1./255,
                                        validation_split=0.1)
        
        train_gen_nopreprocess = imgDataGen_nopreprocess.flow_from_directory(data_dir,
                                target_size=(input_shape[1], input_shape[2]),
                                color_mode='rgb',
                                class_mode='categorical',
                                batch_size=batch_size,
                                shuffle=True,
                                seed=None,
                                subset='training',
                                interpolation='nearest'),
                                #classes = ['cat', 'dog'] if args.train_counterfactual_net else label_map)#['cat', 'dog'])
                                #classes = ['cat'] if args.train_counterfactual_net else label_map)#['cat', 'dog'])
                                #classes = label_map)
        test_gen_nopreprocess  = imgDataGen_nopreprocess.flow_from_directory(data_dir,
                                target_size=(input_shape[1], input_shape[2]),
                                color_mode='rgb',
                                class_mode='categorical',
                                batch_size=batch_size,
                                shuffle=False,
                                seed=None,
                                subset='validation',
                                interpolation='nearest'),
                                #classes = ['cat', 'dog'] if args.train_counterfactual_net else label_map)#['cat', 'dog'])
                                #classes = ['cat'] if args.train_counterfactual_net else label_map)#['cat', 'dog'])
                                #classes = label_map)
        if (args.dataset == 'CUB200' and official_split) or args.dataset == 'BraTS':
            #actual unseen test set
            imgDataGen_official_split = ImageDataGenerator(preprocessing_function = preprocess_input)
            actual_test_gen  = imgDataGen_official_split.flow_from_directory(data_dir_test,
                            target_size=(input_shape[1], input_shape[2]),
                            color_mode='rgb',
                            class_mode='categorical',
                            batch_size=batch_size,
                            shuffle=False,
                            seed=None,
                            #subset='validation',
                            interpolation='nearest')
            imgDataGen_official_split_nopreprocess = ImageDataGenerator(rescale = 1./255)
            actual_test_gen_nopreprocess  = imgDataGen_official_split_nopreprocess.flow_from_directory(data_dir_test,
                            target_size=(input_shape[1], input_shape[2]),
                            color_mode='rgb',
                            class_mode='categorical',
                            batch_size=batch_size,
                            shuffle=False,
                            seed=None,
                            #subset='validation',
                            interpolation='nearest')                            

#%% Trains for 5 epochs.
#if args.train_using_builtin_fit_method and args.fine:
    model.trainable = True
    model.compile(optimizer=optimizers.SGD(lr=0.001/10, momentum = 0.9), 
                  loss=[loss], 
                  metrics=['accuracy'])
    model.summary()
    
    model.load_weights(filepath=weights_path+'/model_transfer_epoch_50.hdf5')
    
    #%%
    save_path=weights_path+'/model_fine_tune.{epoch:02d}-{val_loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only = True)
    callbacks_list = [checkpoint]
    #%%
    if args.dataset == 'cxr1000':
        history = model.fit(train_gen, epochs=30, verbose=1, callbacks=callbacks_list, validation_data=valid_gen, shuffle=True)
    else:
        history = model.fit(train_gen, epochs=50, verbose=1, callbacks=callbacks_list, validation_data=test_gen, shuffle=True)
    
    plt.style.use('seaborn')
    #plt.style.available
    #['fivethirtyeight',
     #'seaborn-pastel',
     #'seaborn-whitegrid',
     #'ggplot',
     #'grayscale']
    
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    #plt.savefig(fname='model_accuracy_'+db+'.png')
    
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    #plt.savefig(fname='model_loss_'+db+'.png')

#%%test accuracy on unseen data
if args.test:
    if (args.dataset == 'CUB200' and official_split) or args.dataset == 'BraTS' or args.dataset == 'NIST':
        #load best weights
        # model.load_weights(filepath=weights_path+'/model_fine_tune.127-1.6321.hdf5')
        # model.load_weights(filepath=weights_path+'/model_fine_tune.19-0.1905.hdf5')
        model.load_weights(filepath=weights_path+'/model_fine_tune_epoch_150.hdf5')
        
        #model.evaluate(actual_test_gen,verbose=1)
             
        pred_probs= model.predict(actual_test_gen,verbose=1)
        
        pred_classes = np.argmax(pred_probs,1)
        #actual_classes = np.argmax(test_gen.classes,1)
        actual_classes = actual_test_gen.classes
        print(confusion_matrix(actual_classes,pred_classes))
        print(classification_report(actual_classes,pred_classes,digits=4)) 
    elif args.dataset == 'cxr1000':
        model.compile(optimizer=optimizers.SGD(lr=0.001/10, momentum = 0.9), 
                  loss=[loss], 
                  metrics=['accuracy'])        
        model.load_weights(filepath=weights_path+'/model_fine_tune_epoch_150.hdf5')
        

        test_loss,test_acc = model.evaluate(test_gen,verbose=1)
        # valid_gen.shuffle=False     
        # pred_probs= model.predict(test_gen,verbose=1)
        # actual_classes = test_gen.labels

        # valid_gen.shuffle=True
        
        # pred_classes = pred_probs.copy()
        # pred_classes[pred_classes>=0.5]=1
        # pred_classes[pred_classes<0.5]=0
        
        
        # pred = np.sum(pred_classes,0)
        # actual = np.sum(actual_classes,0)
        # acc = sum(pred/actual)/len(actual)
        
        # print("acc", acc*100,  "%")

 
    #%% stop execution
    sys.exit()
    
    #%%
if args.test and args.train_using_builtin_fit_method and not args.create_counterfactual_combined:
    model.compile(optimizer=optimizers.SGD(lr=0.01/10, momentum = 0.9), 
                  loss=[categorical_crossentropy], 
                  metrics=['accuracy'])
        
    if (args.dataset == 'CUB200' and official_split) or args.dataset == 'BraTS': test_gen = actual_test_gen
    pred_probs = model.predict(test_gen,verbose=1)
    pred_classes = np.argmax(pred_probs,1)
    #actual_classes = np.argmax(test_gen.y,1)
    actual_classes = test_gen.classes
    print(confusion_matrix(actual_classes,pred_classes))
    print(classification_report(actual_classes,pred_classes,digits=4))
    # print('Test loss:', test_scores[0])
    # print('Test accuracy:', test_scores[1])
    #%% manual training loop
    sys.exit()
"""with or without logits is making a difference in accuracy and loss when compared to model.fit
#why? because i have used softmax as final output layer so the network outputs probabilities instead of logits... therefore from_logits must if False in this case.
#   if i want to apply softmax after getting the output from the network, then i can use from_logits=True to compute the loss """

if args.dataset == 'cxr1000':
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    train_acc_metric = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    test_acc_metric = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
else:
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


optimizer = RMSprop(0.001)

train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
train_loss_metric2 = tf.keras.metrics.Mean(name='train_loss2')

test_loss_metric = tf.keras.metrics.Mean(name='test_loss')


# initialize with some value or load saved/predetermined categories depending on selected method

# k = base_model.get_output_shape_at(0)[3] #number of output filters
# filter_mean_activations_1 = tf.zeros((k,num_classes))*-1 # mean activation per class for each filter... then choose argmax as filter category
# filter_mean_activations_2 = tf.zeros((k,num_classes))*-1

#%%
"""tf.function constructs a callable that executes a TensorFlow graph (tf.Graph) created by trace-compiling the TensorFlow operations in func, effectively executing func as a TensorFlow graph.
#uncomment following line for faster training but it becomes non-debugable and doesnt execute eagerly"""
        
@tf.function 
def train_step(images, labels):
  with tf.GradientTape(persistent=True) as tape: #persistent=False  Boolean controlling whether a persistent gradient tape is created. False by default, which means at most one call can be made to the gradient() method on this object. 
    # training=True is osnly needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    
    if args.full_standard:
        predictions, fmaps = model(images, labels)
    else:
        predictions,x1,x2,target1,target2, raw_map, forward_1 = model(images, labels,training=True)
    

    #predictions = model(images, training=True)
    loss_value = loss_fn(labels, predictions)
    
    total_loss=loss_value
    my_loss=0
    if (args.interpretable and args.loss_compute):
        loss1 = tf.reduce_mean(tf.keras.losses.MAE(target1,x1))
        loss2 = tf.reduce_mean(tf.keras.losses.MAE(target2,x2))
    #loss1 = tf.keras.losses.MAE(target1,x1)
    #loss2 = tf.keras.losses.MAE(target2,x2)
    #print('\nloss1 :', loss1, 'loss2 :', loss2)

        my_loss = 1*(0.5*(loss1+loss2))
        total_loss = my_loss+loss_value
    
    
  gradients = tape.gradient(loss_value, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  if (args.interpretable and args.loss_compute):
        gradients2 = tape.gradient(my_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients2, model.trainable_variables))

  train_loss_metric(loss_value)
  train_loss_metric2(my_loss)

  train_acc_metric(labels, predictions)
  #return x1,x2,target1,target2
@tf.function
def test_step(images, labels, default_fmatrix_test):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  if args.full_standard:
      if args.create_counterfactual_combined:
          predictions, fmaps,_,_,_ = model([images,default_fmatrix_test], training=False)
      else:
         predictions, fmaps = model(images, training=False)
  else:
      predictions,x1,x2,loss1,loss2,raw_map,forward_1 = model(images, training=False)

  loss_value = loss_fn(labels, predictions)

  test_loss_metric(loss_value)
  test_acc_metric(labels, predictions)
  return predictions

#%% for logging output
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = log_path+'/gradient_tape/' + current_time + '/train'
test_log_dir = log_path+'/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


#%%
# Iterate over the batches of the dataset.
#for step, (x_batch_train, y_batch_train) in enumerate(train_gen):
train_gen.reset()
train_gen.batch_index=0
test_gen.reset()
test_gen.batch_index=0
if args.train:
    epochs = 10
    batches=math.ceil(train_gen.n/train_gen.batch_size)
    test_batches=math.ceil(test_gen.n/test_gen.batch_size)

    #print('Running training for %d epocs',epochs)
    for epoch in range(0, epochs):
      #print('Start of epoch %d' % (epoch,))
      
      #for step,(x_batch_train, y_batch_train) in enumerate(dataset):
      with tqdm(total=batches,file=sys.stdout) as progBar:
          for step in range(batches):
            x_batch_train, y_batch_train = next(train_gen)
            #t_x = np.zeros((32,11,11,32))
        
        
            #This method not executing eagerly::
            #loss_value = model.train_on_batch(x_batch_train,[y_batch_train])#, t_x])
            #loss_value = loss_value[1]
            #:::::::::::::::::;
            
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            
            #using tf.function method for faster training
            train_step(x_batch_train, y_batch_train)
            #x1,x2,target1,target2 = train_step(x_batch_train, y_batch_train)
            


            
            # with tf.GradientTape() as tape:
        
            #   # Run the forward pass of the layer.
            #   # The operations that the layer applies
            #   # to its inputs are going to be recorded
            #   # on the GradientTape.
            #   logits,_ = model(x_batch_train, training=True)  # Logits for this minibatch
        
            #   # Compute the loss value for this minibatch.
            #   loss_value = loss_fn(y_batch_train, logits)
        
            # # Use the gradient tape to automatically retrieve
            # # the gradients of the trainable variables with respect to the loss.
            # grads = tape.gradient(loss_value, model.trainable_weights)
        
            # # Run one step of gradient descent by updating
            # # the value of the variables to minimize the loss.
            # optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
            # # Update training metric.
            # train_acc_metric(y_batch_train, logits)
            # train_loss_metric(loss_value)
    
      
            # Log every 200 batches.
            #if step % 200 == 0:
                #print('Training loss (for one batch) at step %s: ' % (step))#, float(loss_value)))
                #print('Seen so far: %s samples' % ((step + 1) * 32))
            progBar.set_description('epoch %d' % (epoch))
            progBar.set_postfix(loss=[train_loss_metric.result().numpy(),train_loss_metric2.result().numpy()], acc=train_acc_metric.result().numpy())
            progBar.update()
            #break
    
          #epoch end
          model.save_weights(filepath=weights_path+'/model_epoch_'+str(epoch)+'.hdf5')
            #save interpretable model specific parameters such as filter classes (class sums and activation sums)
          if (args.interpretable and not args.fixed_classes):
                save_interpretable_parameters(model,weights_path,epoch)
  
           # Display metrics at the end of each epoch.
          train_acc = train_acc_metric.result()
          train_loss = train_loss_metric.result()
          
          #print filter classes
          if (args.interpretable and args.loss_compute and not args.fixed_classes):
              print_filter_classes_1(model)
              print_filter_classes_2(model)

          with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss, step=epoch)
                tf.summary.scalar('accuracy', train_acc, step=epoch)
        
          #print('Training acc over epoch: %s' % (float(train_acc),))
          #print('Training loss over epoch: %s' % (float(train_loss),))
        
          # Reset training metrics at the end of each epoch
          train_acc_metric.reset_states()
          train_loss_metric.reset_states()
          #break
          
      #evaluate model at end of epoch
      evaluate=True
      if evaluate:
          #model.load_weights(weights_path)
          
          #for step,(x_batch_train, y_batch_train) in enumerate(dataset):
         # with tqdm(total=test_batches) as progBar:
          for step in range(test_batches):
            x_batch_test, y_batch_test = next(test_gen)
            
            probs = test_step(x_batch_test, y_batch_test)              
        
           # Display metrics at the end of each epoch.
          test_acc = test_acc_metric.result()
          test_loss = test_loss_metric.result()
          print('\nTest loss:', test_loss.numpy())
          print('Test accuracy:', test_acc.numpy())


elif args.test:
    print('Testing...')

    #model.load_weights(weights_path)
    batches=math.ceil(test_gen.n/test_gen.batch_size)
    default_fmatrix_test = tf.ones((test_gen.batch_size,base_model.output.shape[3]))

    #for step,(x_batch_train, y_batch_train) in enumerate(dataset):
    with tqdm(total=batches,file=sys.stdout,mininterval=10000) as progBar:
        #pass
        for step in range(batches):
          x_batch_test, y_batch_test = next(test_gen)
          default_fmatrix_test = tf.ones((len(x_batch_test),base_model.output.shape[3]))
          
          probs = test_step(x_batch_test, y_batch_test, default_fmatrix_test)
          
          progBar.set_postfix(loss=test_loss_metric.result().numpy(), acc=test_acc_metric.result().numpy(), refresh=False)
          progBar.update()
        #progBar.refresh()    
      
         # Display metrics at the end of each epoch.
        test_acc = test_acc_metric.result()
        test_loss = test_loss_metric.result()
    print('\nTest loss:', test_loss.numpy())
    print('Test accuracy:', test_acc.numpy()) 
#sys.exit()
#%%
# if not args.interpretable:
#     #model.compile(optimizer = optimizers.RMSprop(), loss = tf.keras.losses.BinaryCrossentropy()
#     test_scores = model.evaluate(test_gen, verbose=1)
#     print('Test loss:', test_scores[0])
#     print('Test accuracy:', test_scores[1])


#%% manual test
# pred_probs,_= model.predict(test_gen,verbose=1)

# pred_classes = np.argmax(pred_probs,1)
# actual_classes = np.argmax(test_gen.y,1)
# print(confusion_matrix(actual_classes,pred_classes))
# print(classification_report(actual_classes,pred_classes))
#print('Test loss:', test_scores[0])
#print('Test accuracy:', test_scores[1])
#%%
#train/test base model only
if not args.create_counterfactual_combined:
    sys.exit()
#%% train counterfactual generation network
#args.train_counterfactual_net=False
@tf.custom_gradient
def custom_op(x):
    result = masking_layer(x) # do forward computation
    def custom_grad(dy):
        grad = 1.0 # compute gradient
        return grad
    return result, custom_grad

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomLayer, self).__init__()

    def call(self, x):
        return custom_op(x)  # you don't need to explicitly define the custom gradient
                             # as long as you registered it with the previous method

def masking_layer(tensor):
    a = tf.keras.backend.stop_gradient(tf.where(tensor>=0.5,tf.ones_like(tensor),tf.zeros_like(tensor)))
    return a


# class PN_add_layer(tf.keras.layers.Layer):
#     def __init__(self, units=128, input_dim=128):
#         super(PN_add_layer, self).__init__()
#         w_init = tf.random_normal_initializer()
#         self.w = tf.Variable(
#             initial_value=w_init(shape=(input_dim, units), dtype="float32"),
#             trainable=True,
#         )
#         b_init = tf.zeros_initializer()
#         self.b = tf.Variable(
#             initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
#         )

#     def call(self, inputs):
#         return tf.add(inputs, self.w)

class PN_add_layer(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(PN_add_layer, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim), initializer="ones", trainable=True#zeros#random_normal
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True,name='sss')

    def call(self, inputs):
        #return tf.math.multiply(inputs, self.w) + self.b# Need to perform simple element-wise multiplication
        #return tf.subtract(1.,tf.math.multiply(inputs, self.w))#need to ensure w is -negataive always so that we only add and not subtract the magnitudes
        return tf.math.multiply(inputs, self.w)# + self.b#need to ensure w is -negataive always so that we only add and not subtract the magnitudes
        #return tf.matmul(inputs, self.w) + self.b #need to ensure w is -negataive always so that we only add and not subtract the magnitudes


#sigmoid = tf.convert_to_tensor(np.random.rand(512))
#filter_map = masking_layer(sigmoid)
if (args.train_counterfactual_net or args.load_counterfactual_net):
    num_filters = model.output[1].shape[3]
    model.trainable = False
    
    if args.model == 'VGG16/' or args.model == 'myCNN/':
        x =  MaxPool2D()(base_model.output)
    elif args.model == 'resnet50/':
        x =  base_model.output
    elif args.model == 'efficientnet/':
        x =  base_model.output
    mean_fmap = GlobalAveragePooling2D()(x)
    
    if args.counterfactual_PP:
        x = Dense(num_filters,activation='sigmoid')(mean_fmap)#kernel_regularizer='l1' #,activity_regularizer='l1'
    else:
        x = Dense(num_filters,activation='relu')(mean_fmap)
    #x = tf.keras.layers.Lambda(masking_layer)(x)
    #x = CustomLayer()(x)
    #skipping gradients
    #https://stackoverflow.com/questions/39048984/tensorflow-how-to-write-op-with-gradient-in-python
    
    #custom layer with custom gradients
    #https://stackoverflow.com/questions/56657993/how-to-create-a-keras-layer-with-a-custom-gradient-in-tf2-0/56658149
    
    thresh=0.5
    PP_filter_matrix = tf.keras.layers.ThresholdedReLU(theta=thresh)(x)
    
    if not args.counterfactual_PP: #for PNs
    
        PN_layer = PN_add_layer(PP_filter_matrix.shape[1],input_dim=PP_filter_matrix.shape[1])(PP_filter_matrix)
        PN_layer = tf.keras.layers.ReLU()(PN_layer)
        
        
        #PN_layer = Dense(num_filters,activation='relu')(PP_filter_matrix)

    
    
    if args.counterfactual_PP:
        counterfactual_generator = tf.keras.Model(inputs=base_model.input, outputs= [PP_filter_matrix],name='counterfactual_model')
    else:
        counterfactual_generator = tf.keras.Model(inputs=base_model.input, outputs= [x],name='counterfactual_model')
        counterfactual_generator.summary()

    #counterfactual_generator = tf.keras.Model(inputs=model.input[0], outputs= x,name='counterfactual_model')
    

    if args.train_counterfactual_net:
        cf_epochs = args.cfe_epochs #100 #100 for MNIST, 200 for CUB
        L1_weight = args.l1_weight #2 for MNIST, 2,4,6 for CUB (default 4?)
        for_class = args.alter_class #0 #0-9 for MNIST, 8,9s,10,11 for CUB (default 9) or 0,1,2,3 for subset training data case
        print("threshold: ", thresh)
        print("l1 weight: ", L1_weight)
        print("training CF model for alter class: ",label_map[for_class])
        #TODO: pass parameters using just args
        combined, generator = train_counterfactual_net(model,weights_path, counterfactual_generator, train_gen,args.test_counterfactual_net, args.resume_counterfactual_net,epochs=cf_epochs,L1_weight=L1_weight,for_class=for_class,label_map=label_map,logging=logging,args=args) 
        #if logging: sys.stdout.close()        
        #sys.modules[__name__].__dict__.clear()
        #os._exit(00)
        from IPython import get_ipython
        ipython = get_ipython()
        ipython.magic("reset -f")
        sys.exit()
    else:
        #counterfactual_generator.load_weights(filepath=weights_path+'/counterfactual_generator_model.hdf5')
        
        model.trainable = False
        img = tf.keras.Input(shape=model.input_shape[0][1:4])
        if args.counterfactual_PP:
            fmatrix = counterfactual_generator(img)
        else:
            #fmatrix,PN_add = counterfactual_generator(img)
            PN_add = counterfactual_generator(img)
            fmatrix = PN_add
        
        alter_prediction,fmaps,mean_fmap, modified_mean_fmap_activations,pre_softmax = model([img,fmatrix])
        
        if args.counterfactual_PP:
            combined = tf.keras.Model(inputs=img, outputs=[alter_prediction,fmatrix,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax])
        else:
            combined = tf.keras.Model(inputs=img, outputs=[alter_prediction,fmatrix,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax,PN_add])
        #combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        #combined.summary()
        
        if args.counterfactual_PP:
            mode = '' 
            print("Loading CF model for PPs")
        else:
            mode = 'PN_'
            print("Loading CF model for PNs")
        combined.load_weights(filepath=weights_path+'/'+mode+'counterfactual_combined_model_fixed_'+str(label_map[args.alter_class])+'_alter_class.hdf5')

#%%
W = model.weights[-2]
act_threshold=-0.15
plt.plot(W.numpy()), plt.title('Weights'), plt.legend(['cat','dog']),
plt.hlines(act_threshold,xmin=0,xmax=512,colors='r'),plt.show()

important_filter_weights_1 = np.where(W[:,0]<=act_threshold)
important_filter_weights_2 = np.where(W[:,1]<=act_threshold)
#%%      
  
# tf.keras.utils.plot_model(
#     model, to_file='model.png', show_shapes=False, show_layer_names=True,
#     rankdir='TB', expand_nested=False, dpi=96
# )

# tf.keras.utils.plot_model(
#     counterfactual_generator, to_file='counterfactual.png', show_shapes=False, show_layer_names=True,
#     rankdir='TB', expand_nested=False, dpi=96
# )

# tf.keras.utils.plot_model(
#     combined, to_file='combined.png', show_shapes=False, show_layer_names=True,
#     rankdir='TB', expand_nested=False, dpi=96
# )
#%% find_filter_class

if args.find_filter_class and args.full_standard and args.resume:
    
    class_activation_sums, class_img_count = find_filter_class(model,train_gen)    

    #find the mean activations corresponding to classes for each filter
    filter_means = tf.zeros(class_activation_sums.shape,dtype=tf.float32)
    ## filter categories assigned, now compute loss image/template pair in the batch
    filter_means = tf.math.divide_no_nan(class_activation_sums,tf.cast(class_img_count,tf.float64))
    filter_class = tf.argmax(filter_means,1,output_type=tf.dtypes.int32)            

    plt.plot(filter_means)
    #np.save(file=weights_path+'/class_activation_sums.npy',arr=class_activation_sums)
    #np.save(file=weights_path+'/class_img_count.npy',arr=class_img_count)
    
#%% fitler activation statistical analysis
if False:
    class_activation_sums = np.load(file=weights_path+'/class_activation_sums.npy')
    class_img_count = np.load(file=weights_path+'/class_img_count.npy')
    
    filter_means = tf.math.divide_no_nan(class_activation_sums,tf.cast(class_img_count,tf.float64))
    filter_class = tf.argmax(filter_means,1,output_type=tf.dtypes.int32)            


    act_threshold = 10

    plt.plot(filter_means), plt.title('filter_means'), plt.legend(['cat','dog'])
    plt.hlines(act_threshold,xmin=0,xmax=512,colors='r'),plt.show()
    
    act_diff = (filter_means[:,0]-filter_means[:,1])
    plt.plot(act_diff), plt.title('activation differences'),plt.show()
    
    
    important_filters_1 = np.where(filter_means[:,0]>=act_threshold)
    important_filters_2 = np.where(filter_means[:,1]>=act_threshold)

    print('important_filters for class 0:', important_filters_1)
    print('important_filters for class 1:', important_filters_2)

#%% argmax class
    class_filters = np.argmax(filter_means,axis=1)
    counts = plt.hist(class_filters), plt.show()
    #print(counts[0])
    
    cat_filters = np.where(class_filters==0)
    dog_filters = np.where(class_filters==1)
    
    top_n = 5# pick top n filters for each class
    
    top_cat = filter_means[:,0].numpy().argsort()[:-top_n]# switch off... other than top n
    top_dog = filter_means[:,1].numpy().argsort()[:-top_n]# switch off... other than top n

#%%adversarial example generation code

def input_derivative(net, x, y):
    """ Calculate derivatives wrt the inputs"""  
    #nabla_b = [np.zeros(b.shape) for b in net.biases]
    #nabla_w = [np.zeros(w.shape) for w in net.weights]
    
    # feedforward
    # activation = x
    # activations = [x] # list to store all the activations, layer by layer
    # zs = [] # list to store all the z vectors, layer by layer
    # for bias, weight in zip(net_biases, net_weights):
    #     z = np.dot(weight, activation)+bias
    #     zs.append(z)
    #     activation = sigmoid(z)
    #     activations.append(activation)
    
    #net.trainable = True
    #my feedforward
    with tf.GradientTape(persistent=False) as tape: #persistent=False  Boolean controlling whether a persistent gradient tape is created. False by default, which means at most one call can be made to the gradient() method on this object. 
        tape.watch(x)
        pred_probs = net(x, training=False)#with eager
        loss = loss_fn(y.T, pred_probs)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, x)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)


    #net.trainable = False
    
    # #backward pass
    # delta = net.cost_derivative(activations[-1], y) * \
    #     sigmoid_prime(zs[-1])
    # nabla_b[-1] = delta
    # nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    # for l in range(2, net.num_layers):
    #     z = zs[-l]
    #     sp = sigmoid_prime(z)
    #     delta = np.dot(net.weights[-l+1].transpose(), delta) * sp
    #     nabla_b[-l] = delta
    #     nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
    # Return derivatives WRT to input
    return signed_grad, loss#net.weights[0].T.dot(delta) #gradients

def sneaky_adversarial(net, n, x_target, steps, eta, lam=.05):
    """
    net : network object
        neural network instance to use
    n : integer
        our goal label (just an int, the function transforms it into a one-hot vector)
    x_target : numpy vector
        our goal image for the adversarial example
    steps : integer
        number of steps for gradient descent
    eta : float
        step size for gradient descent
    lam : float
        lambda, our regularization parameter. Default is .05
    """
    # Set the goal output
    goal = np.zeros((num_classes, 1))
    goal[n] = 1

    # Create a random image to initialize gradient descent with
    x = np.random.normal(.5, .3, (x_target.shape))
    
    #feedforward
    #pred_probs,fmaps,mean_fmaps,_ ,pre_softmax= model([np.expand_dims(x_batch_test[img_ind],0),np.expand_dims(default_fmatrix[0],0)], training=False)#with eager

    # Gradient descent on the input
    x=tf.convert_to_tensor(np.expand_dims(x,0))
    for i in range(steps):
        # Calculate the derivative
        derivative, loss = input_derivative(net,x,goal)
        print("adversarial attack training step: ", i, "loss: ",str(loss.numpy()))

        # The GD update on x, with an added penalty to the cost function
        # ONLY CHANGE IS RIGHT HERE!!!
        x -= eta * (derivative + lam * (x - x_target))
        #x -= eta * (derivative + 0)

    return x

# Wrapper function
def sneaky_generate(n, m):
    """
    n: int , the target number to match
    m: example image 
    """
    
    
    # Hardcode the parameters for the wrapper function
    a = sneaky_adversarial(model_vgg_original, n, m, steps=100, eta=1)
    #x = np.round(net.feedforward(a), 2)
    a_post_process = restore_original_image_from_array(a.numpy().squeeze())
    
    pred_probs= model_vgg_original(a, training=False)#with eager
    
    m_post_process = restore_original_image_from_array(m.squeeze())

    #print('\nOriginal image: ')
    plt.imshow(m_post_process.astype('uint8'))
    plt.show()
    

    
    #print('Adversarial Example: ')
    
    plt.imshow(a_post_process.astype('uint8'))
    plt.show()
    
    #print('Network Prediction: ' + str(label_map[np.argmax(pred_probs)]) + '\n')
    
    #print('Adversarial data: ')
    
    plt.imshow((m_post_process-a_post_process).astype('uint8')), plt.show()
    
    return a


#%% analyze_filter_importance
explainer = GradCAM()

generate_adversarial = False

if generate_adversarial:
    x =  MaxPool2D()(base_model.output)
    mean_fmap = GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(mean_fmap)
    x = Dense(num_classes,activation='softmax')(x)
    model_vgg_original = tf.keras.Model(inputs=base_model.input, outputs= [x])
    model_vgg_original.summary()
    
    model_vgg_original.load_weights(filepath=weights_path+'/model_fine_tune_epoch_150.hdf5')
    print("model_vgg_original loaded")

#['bird',  'cat', 'cow', 'dog', 'horse', 'sheep']
predictive_counterfactual_method = True# else statistical method
misclassification_analysis = True
if True:
    class_for_analysis = args.analysis_class#9#9 170#np.random.randint(200)#23#11 #cat for VOC dataset
    alter_class=args.alter_class
    print ('class for analysis: ', label_map[class_for_analysis])
    print ('alter class: ', label_map[alter_class])
    print ('class 2: ', label_map[args.alter_class_2])
    
    weights_alter_class = W[:,alter_class]
    weights_class_2 = W[:,args.alter_class_2]    
    plt.plot(weights_alter_class),plt.title("weight alter class "+str(alter_class)),plt.show()
    plt.plot(weights_class_2),plt.title("weight class "+str(args.alter_class_2)),plt.show()
    
    if args.dataset == 'CUB200' or args.dataset == 'BraTS' or args.dataset == 'NIST': 
        test_gen =train_gen if args.find_global_filters else actual_test_gen# train_gen#actual_test_gen
        test_gen_nopreprocess = train_gen_nopreprocess if args.find_global_filters else actual_test_gen_nopreprocess #train_gen_nopreprocess[0]#actual_test_gen_nopreprocess
        print("using traingen gen data") if args.find_global_filters else print("using testgen gen data")
    # gen=test_gen
    gen=test_gen_user_eval
    test_gen_nopreprocess = test_gen_user_eval_nopreprocess
    
    batches=math.ceil(gen.n/gen.batch_size)
    
    #print('important_filters for class 0:', important_filters_1)
    #print('top pos filters', pos_filters)
    if args.model =='myCNN/': test_gen_nopreprocess = gen
    
    
    filter_histogram_cf = tf.zeros(fmatrix[0].shape[0])
    filter_magnitude_cf = tf.zeros(fmatrix[0].shape[0])
    filter_histogram_default = tf.zeros(fmatrix[0].shape[0])
    filter_magnitude_default = tf.zeros(fmatrix[0].shape[0])
    filter_sum = 0


    gen.reset() #resets batch index to 0
    test_gen_nopreprocess.reset()
    local_misclassifications = 0
    pred_confidence_alter = []
    pred_confidence_orig = []
    img_count=0
    alter_class_images_count = np.sum(gen.classes==alter_class)
    alter_class_starting_batch = np.floor(np.where(gen.labels==args.alter_class)[0][0]/gen.batch_size).astype(np.int32)
    index_reached=0
    for k in range(batches):

        if args.find_global_filters:
            if k < alter_class_starting_batch:
                sys.stdout.write("\rbatch %i of %i" % (k, batches))
                continue
            else:
                x_batch_test,y_batch_test = next(gen)                
                gen.batch_index = k

            
        x_batch_test,y_batch_test = next(gen)
        
        if args.find_global_filters:
            if gen.batch_index < alter_class_starting_batch and gen.batch_index >0:
                continue

        
        if args.model =='myCNN/': x_batch_test_nopreprocess = x_batch_test
        else: x_batch_test_nopreprocess,_ = next(test_gen_nopreprocess)
        #print("batch ",k," of ", batches)
        sys.stdout.write("\rbatch %i of %i" % (k, batches))
        sys.stdout.flush()
        
        statistical_analysis = False #global statisitcs for all classes
        if statistical_analysis:
            #compute histpgram of activated filters
            #keep track of activation magnitude
            default_fmatrix = tf.ones((x_batch_test.shape[0],base_model.output.shape[3]))
            pred_probs,fmaps,mean_fmaps,_ ,pre_softmax= model([x_batch_test,default_fmatrix], training=False)#with eager
            alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(x_batch_test)                
            
            t_fmatrix = fmatrix.numpy()
            for i in tf.where(fmatrix>0):
                t_fmatrix[i]=1.0
            t_fmatrix = tf.convert_to_tensor(t_fmatrix)
            alter_probs, c_fmaps, c_mean_fmap, c_modified_mean_fmap_activations,alter_pre_softmax = model([x_batch_test,t_fmatrix])#with eager
            
            
            for i in range(fmatrix.shape[0]): 
                filter_histogram_cf += t_fmatrix[i] 
                filter_magnitude_cf += c_modified_mean_fmap_activations[i]
            # filter_histogram_default += tf.zeros_like(fmatrix[0])
            # filter_magnitude_default += tf.zeros_like(fmatrix[0])
            filter_sum += fmatrix.shape[0]
            
            if k==batches-1:
                print("\nfinished")
                plt.plot(filter_histogram_cf), plt.show()
                plt.plot(filter_magnitude_cf), plt.show()
                plt.plot(filter_magnitude_cf/(filter_histogram_cf+0.00001)), plt.show()
                
                mName = args.model[:-1]
                #plt.ylim([0, np.max(c_mean_fmap)+1]), 
                plt.plot(filter_histogram_cf), plt.savefig(fname="./figs_for_paper/"+mName+"_filter_histogram_cf_"+str(alter_class)+"_all_test.png", dpi=None, bbox_inches = 'tight'), plt.show()
                #plt.plot(filter_magnitude_cf/(filter_histogram_cf+0.00001)), plt.savefig(fname="./figs_for_paper/"+mName+"_avg_filter_magnitude_cf_"+str(class_for_analysis)+"_all_test.png", dpi=None, bbox_inches = 'tight'), plt.show()
                #plt.plot(filter_magnitude_cf), plt.savefig(fname="./figs_for_paper/"+mName+"_filter_magnitude_cf_"+str(class_for_analysis)+"_all_test.png", dpi=None, bbox_inches = 'tight'), plt.show()
                plt.plot(filter_magnitude_cf/max(filter_histogram_cf)), plt.savefig(fname="./figs_for_paper/"+mName+"_normalized_filter_magnitude_cf_"+str(alter_class)+"_all_test.png", dpi=None, bbox_inches = 'tight'), plt.show()
                
                np.save(file= "./figs_for_paper/"+mName+"_filter_histogram_cf_"+str(alter_class)+"_all_test.np",arr=filter_histogram_cf)
                np.save(file= "./figs_for_paper/"+mName+"_normalized_filter_magnitude_cf_"+str(alter_class)+"_all_test.np", arr=filter_magnitude_cf)
                
                sys.exit()
            continue


        for i in range (len(x_batch_test)):
            img_ind = i#3
            
            if gen.batch_index==0:
                actual_img_ind = i + (batches-1)*gen.batch_size
            else:
                actual_img_ind = i + (gen.batch_index-1)*gen.batch_size
            
            if misclassification_analysis and False:
                if actual_img_ind != 240:#240, 241 wrong for class 9; 655 for 25
                    print('skipping img_ind:',actual_img_ind)
                    continue
                    

            y_gt = y_batch_test[img_ind]
            alter_class_list = np.array([25,108,9,9,9,9,170,170,125,125])
            
            ######################
            class_for_analysis = np.argmax(y_gt)
            alter_class = class_for_analysis#alter_class_list[img_ind]
            
            args.alter_class = class_for_analysis
            ######################

             #skip other class        
            if class_for_analysis==np.argmax(y_gt):
                pass
                print('\n\nimg_ind:',actual_img_ind)
            else:
                pass
                # continue
            
            if args.counterfactual_PP:
                mode = '' 
                print("Loading CF model for PPs")
            else:
                mode = 'PN_'
                print("Loading CF model for PNs")
            combined.load_weights(filepath=weights_path+'/'+mode+'counterfactual_combined_model_fixed_'+str(label_map[class_for_analysis])+'_alter_class.hdf5')

            # statistical_analysis = True #global statistics for alter class images only
            if args.find_global_filters:
                # assert(args.find_global_filters==True)
                #compute histpgram of activated filters
                #keep track of activation magnitude
                if sum(y_batch_test[:,args.alter_class])<len(x_batch_test):
                    #process sequentially
                    alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(np.expand_dims(x_batch_test[img_ind],0))                
                
                    t_fmatrix = fmatrix.numpy()
                    for i in tf.where(fmatrix>0):
                        t_fmatrix[i]=1.0
                    t_fmatrix = tf.convert_to_tensor(t_fmatrix)
                    alter_probs, c_fmaps, c_mean_fmap, c_modified_mean_fmap_activations,alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),t_fmatrix])#with eager
                    
                    # print('thresholded counterfactual')
                    # print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')
                    # print( 'alter class: ',label_map[alter_class], '  prob: ',alter_probs[0][alter_class].numpy()*100,'%')
                    # print('alter_pre_softmax: ',alter_pre_softmax[0][0:10])
                    
                    filter_histogram_cf += t_fmatrix[0]
                    filter_magnitude_cf += c_modified_mean_fmap_activations[0]
                    # filter_histogram_default += tf.zeros_like(fmatrix[0])
                    # filter_magnitude_default += tf.zeros_like(fmatrix[0])
                    filter_sum += 1
                    print("image_count",filter_sum)
                    # sys.stdout.write("\rimage_count %i" % (filter_sum))
                    # sys.stderr.flush()
                else:
                    #process in batch
                    alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(x_batch_test)                
                
                    t_fmatrix = fmatrix.numpy()
                    for i in tf.where(fmatrix>0):
                        t_fmatrix[tuple(i)]=1.0
                    t_fmatrix = tf.convert_to_tensor(t_fmatrix)
                    alter_probs, c_fmaps, c_mean_fmap, c_modified_mean_fmap_activations,alter_pre_softmax = model([x_batch_test,t_fmatrix])#with eager
                    
                    # print('thresholded counterfactual')
                    # print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')
                    # print( 'alter class: ',label_map[alter_class], '  prob: ',alter_probs[0][alter_class].numpy()*100,'%')
                    # print('alter_pre_softmax: ',alter_pre_softmax[0][0:10])
                    
                    for i in range(len(t_fmatrix)):
                        filter_histogram_cf += t_fmatrix[i] 
                        filter_magnitude_cf += c_modified_mean_fmap_activations[i]
                    
                    # filter_histogram_default += tf.zeros_like(fmatrix[0])
                    # filter_magnitude_default += tf.zeros_like(fmatrix[0])
                    filter_sum += len(t_fmatrix)
                    print("image_count",filter_sum)
                    #only break if its not the last batch
                    if k == batches-1:
                        pass
                    else:
                        break
                    # sys.stdout.write("\rimage_count %i" % (filter_sum))
                    # sys.stderr.flush()
                
                
                if args.dataset=='mnist':
                    test_image_count = [ 980,
                                        1135,
                                        1032,
                                        1010,
                                         982,
                                         892,
                                         958,
                                        1028,
                                         974,
                                        1009]
                else:
                    # test_image_count = 30
                    test_image_count = alter_class_images_count
                if filter_sum==alter_class_images_count:#test_image_count[class_for_analysis]:
                    
                    print("\nfinished")
                    # plt.plot(filter_histogram_cf), plt.show()
                    # plt.plot(filter_magnitude_cf), plt.show()
                   # plt.plot(filter_magnitude_cf/(filter_histogram_cf+0.00001)), plt.show()
                    
                    mName = args.model[:-1]+'_'+args.dataset
                    #plt.ylim([0, np.max(c_mean_fmap)+1]), 
                    plt.plot(filter_histogram_cf), plt.ylim([0, np.max(filter_histogram_cf)+1]),plt.xlabel("Filter number"),plt.ylabel("Filter activation count"), plt.savefig(fname="./figs_for_paper/"+mName+"_filter_histogram_cf_alter_class_"+str(alter_class)+"_"+str(class_for_analysis)+"_train_set.png", dpi=300, bbox_inches = 'tight'), plt.show()
                    plt.plot(filter_magnitude_cf/max(filter_histogram_cf)),plt.xlabel("Filter number"),plt.ylabel("Avg. activation magnitude"), plt.ylim([0, np.max(filter_magnitude_cf/np.max(filter_histogram_cf))+1]), plt.savefig(fname="./figs_for_paper/"+mName+"_normalized_filter_magnitude_cf_alter_class_"+str(alter_class)+"_"+str(class_for_analysis)+"_train_set.png", dpi=300, bbox_inches = 'tight'), plt.show()
                    #plt.plot(filter_magnitude_cf/max(filter_histogram_cf)), plt.ylim([0, np.max(filter_magnitude_cf/max(filter_histogram_cf))+1]), plt.show()
                    
                    np.save(file= "./figs_for_paper/"+mName+"_filter_histogram_cf_"+str(alter_class)+"_train_set.np",arr=filter_histogram_cf)
                    np.save(file= "./figs_for_paper/"+mName+"_normalized_filter_magnitude_cf_"+str(alter_class)+"_train_set.np", arr=filter_magnitude_cf)
                    
                    
                    #######################
                    #thresholded stats
                    
                    
                    filter_histogram_cf_thresholded = np.zeros_like(filter_histogram_cf)
                    thresh=9
                    filter_histogram_cf_thresholded[filter_histogram_cf>thresh] = filter_histogram_cf[filter_histogram_cf>thresh]
                    
                    #plt.plot(filter_histogram_cf_thresholded), plt.ylim([0, np.max(filter_histogram_cf_thresholded)+1]),plt.savefig(fname="./figs_for_paper/"+mName+"_filter_histogram_thresholded_"+str(thresh)+"_cf_alter_class_"+str(alter_class)+"_"+str(class_for_analysis)+"_test_set.png", dpi=None, bbox_inches = 'tight'), plt.show()
                    #TODO::plt.plot(filter_magnitude_cf/max(filter_histogram_cf)), plt.ylim([0, np.max(filter_magnitude_cf/np.max(filter_histogram_cf))+1]), plt.savefig(fname="./figs_for_paper/"+mName+"_normalized_filter_magnitude_thresholded_"+str(thresh)+"_cf_alter_class_"+str(alter_class)+"_"+str(class_for_analysis)+"_train_set.png", dpi=None, bbox_inches = 'tight'), plt.show()
                    
                    sys.exit()
                continue
            
            
            #model.load_weights(filepath=weights_path+'/model.hdf5')  
            pred_probs,fmaps,mean_fmaps,_ ,pre_softmax= model([np.expand_dims(x_batch_test[img_ind],0),np.expand_dims(default_fmatrix[0],0)], training=False)#with eager
            #pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
            print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
            print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%')
            #print('pre_softmax: ',pre_softmax[0][0:10])
            

            
            if generate_adversarial:
                # sneaky_generate(target label, target digit)
                pred_probs= model_vgg_original([np.expand_dims(x_batch_test[img_ind],0)], training=False)#with eager
                #pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
                print('\npredicted_vgg_original: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
                print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%')
                
                adv_ex = sneaky_generate(args.alter_class, x_batch_test[img_ind])
                
                
                pred_probs,fmaps,mean_fmaps,_ ,pre_softmax= model([adv_ex, np.expand_dims(default_fmatrix[0],0)], training=False)#with eager
                print('\nadversarial attack \npredicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
                print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%')
                #print('pre_softmax: ',pre_softmax[0][0:10])
                
                x_batch_test[img_ind] = adv_ex.numpy().reshape((224,224,3))
            
            #wrong prediction
            if np.argmax(pred_probs) != np.argmax(y_gt) and True:
                print("wrong prediction")
                # incorrect_class=np.argmax(pred_probs)
                print("skipping wrong prediction")
                continue
            else:
                pass
                # print("skipping correct prediction")
                # continue
            
           # #skip high confidence predictions
            skip_high_confidence = False
            if skip_high_confidence:
                if pred_probs[0][np.argmax(y_gt)]>0.9:
                    print("skipping high confidence prediction")
                    continue
            
            skip_low_confidence_alter = False
            if skip_low_confidence_alter:
                alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(np.expand_dims(x_batch_test[img_ind],0))                
                if alter_prediction[0][alter_class]<0.9:
                    print("skipping low alter prediction")
                    continue
            
            top_k_preds = True if args.dataset == 'CUB200'  else False
            if top_k_preds:
                k=3
                ind = np.argpartition(pred_probs[0], -k)[-k:]
                ind=ind[np.argsort(pred_probs[0].numpy()[ind])]
                
                for i in range(k):
                    print('top ',str(i+1)+' predicted: ',label_map[ind[k-1-i]], ' with prob: ',pred_probs[0][ind[k-1-i]].numpy()*100,'%')

                
            gradcam = True
            if gradcam:
                output_orig,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,np.argmax(pred_probs),image_nopreprocessed=np.expand_dims(x_batch_test_nopreprocess[img_ind],0),fmatrix=default_fmatrix)
                
                plt.imshow(output_orig), plt.axis('off'), plt.title('original prediction')
                plt.show()
                
                output_gradcam_alter,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,alter_class,image_nopreprocessed=np.expand_dims(x_batch_test_nopreprocess[img_ind],0),fmatrix=default_fmatrix)
                plt.imshow(output_gradcam_alter), plt.axis('off'), plt.title('GradCAM alter prediction')
                plt.show()
            original_mean_fmap_activations = GlobalAveragePooling2D()(fmaps)
            #plt.plot(original_mean_fmap_activations[0]), plt.title('original_mean_fmap_activations'),plt.show()
            plt.plot(mean_fmaps[0]), plt.ylim([0, np.max(mean_fmaps)+1]), plt.title('mean_fmaps'),plt.show()

            
            if not predictive_counterfactual_method:
                class_filters_off = 1
                if class_filters_off==0:
                    print('removing cat filters')
                    filters_off =cat_filters[0]
                else:
                    print('removing dog filters')
                    filters_off =dog_filters[0]
                
                filters_off = top_dog
                #give indexes of filters to switch off
                #important_filters_1[0],dog_filters #cat_filters
                modified_model, modified_fmaps = check_histogram_top_filter_result(model,filters_off,x_batch_test[img_ind],y_gt,label_map,args)
                
                modified_mean_fmap_activations = GlobalAveragePooling2D()(modified_fmaps)
                plt.plot(modified_mean_fmap_activations[0]), plt.title('modified_mean_fmap_activations'), plt.show()
            else:
                #combined.load_weights(filepath=weights_path+'/counterfactual_combined_model_fixed_'+str(label_map[incorrect_class])+'_alter_class.hdf5')
                
                #fmatrix = counterfactual_generator(np.expand_dims(x_batch_test[img_ind],0))
                if args.counterfactual_PP:
                    alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(np.expand_dims(x_batch_test[img_ind],0))
                    filters_off = fmatrix
                else:
                    alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax,PN_add = combined(np.expand_dims(x_batch_test[img_ind],0))
                    filters_off = PN_add
                
                alter_probs, c_fmaps, c_mean_fmap, c_modified_mean_fmap_activations,alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),filters_off])#with eager
                
                #print('\ncounterfactual')
                #print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')
                #print( 'alter class: ',label_map[alter_class], '  prob: ',alter_probs[0][alter_class].numpy()*100,'%')
                #print('alter_pre_softmax: ',alter_pre_softmax[0][0:10])

                #modified_model, modified_fmaps = check_histogram_top_filter_result(model,filters_off,x_batch_test[img_ind],y_gt,label_map,args)
                #plt.plot(fmatrix[0]), plt.title('fmatrix'), plt.show()

                #modified_mean_fmap_activations = GlobalAveragePooling2D()(modified_fmaps)
                #plt.plot(modified_mean_fmap_activations[0]), plt.title('modified_mean_fmap_activations'), plt.show()
                
            #np.where(modified_mean_fmap_activations[0]>=8)
            #model.load_weights(filepath=weights_path+'/model.hdf5')    
            #print('pos')
            #check_histogram_top_filter_result(model,pos_filters,x_batch_test[img_ind],y_gt,label_map,args)
            
            gradcam=False
            if gradcam:
                #x_batch_test_nopreprocess
                #x_batch_test
                
                if not predictive_counterfactual_method:
                    output = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),modified_model,np.argmin(y_batch_test[img_ind]),image_nopreprocessed=np.expand_dims(x_batch_test_nopreprocess[img_ind],0))
                else:
                    output_cf,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,alter_class,image_nopreprocessed=np.expand_dims(x_batch_test_nopreprocess[img_ind],0),fmatrix=fmatrix,image_weight=0.7)#np.argmin(y_batch_test[img_ind])
                
                plt.imshow(output_cf), plt.axis('off'), plt.title('modified prediction')
                plt.show()
                plt.imshow(np.squeeze(x_batch_test_nopreprocess[img_ind])), plt.axis('off'), plt.title('original image')
                plt.show()
                y_gt
            
            #%% subfigures
            # fig, axs = plt.subplots(2, 2,figsize=(15,10))
            # axs[0, 0].imshow(output_orig), axs[0, 0].axis('off'), axs[0, 0].set_title('original prediction')
            # axs[0, 1].imshow(output_cf), axs[0, 1].axis('off'), axs[0, 1].set_title('modified prediction')
            # axs[1, 0].plot(mean_fmaps[0]),axs[1,0].set_ylim([0, np.max(mean_fmaps)+1]), axs[1, 0].set_title('mean_fmaps')
            # axs[1, 1].plot(modified_mean_fmap_activations[0]),axs[1,1].set_ylim([0, np.max(mean_fmaps)+1]), axs[1, 1].set_title('modified_mean_fmap_activations')
            # plt.show()            
            
            #%%
            apply_thresh = True
            if apply_thresh:
                t_fmatrix = filters_off.numpy()
                if args.counterfactual_PP:
                    for i in tf.where(filters_off>0):
                        t_fmatrix[tuple(i)]=1.0
                    t_fmatrix = tf.convert_to_tensor(t_fmatrix)
                alter_probs, c_fmaps, c_mean_fmap, c_modified_mean_fmap_activations,alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),t_fmatrix])#with eager
                
                print('\nthresholded counterfactual')
                print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')
                print( 'alter class: ',label_map[alter_class], '  prob: ',alter_probs[0][alter_class].numpy()*100,'%')
               # print('alter_pre_softmax: ',alter_pre_softmax[0][0:10])
                if top_k_preds:
                    k=3
                    ind = np.argpartition(alter_probs[0], -k)[-k:]
                    ind = ind[np.argsort(alter_probs[0].numpy()[ind])]
                    
                    for i in range(k):
                        print('top ',str(i+1)+' predicted: ',label_map[ind[k-1-i]], ' with prob: ',alter_probs[0][ind[k-1-i]].numpy()*100,'%')
               
                #modified_model, modified_fmaps = check_histogram_top_filter_result(model,filters_off,x_batch_test[img_ind],y_gt,label_map,args)
                #plt.plot(fmatrix[0]), plt.title('fmatrix'), plt.show()

                #modified_mean_fmap_activations = GlobalAveragePooling2D()(modified_fmaps)
                plt.plot(c_modified_mean_fmap_activations[0]),plt.ylim([0, np.max(c_mean_fmap)+1]), plt.title('thresh_mean_fmap_activations'), plt.show()
                
                output_cf,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,alter_class,image_nopreprocessed=np.expand_dims(x_batch_test_nopreprocess[img_ind],0),fmatrix=t_fmatrix,image_weight=0.7)#np.argmin(y_batch_test[img_ind])
                
                plt.imshow(output_cf), plt.axis('off'), plt.title('thresh prediction')
                plt.show()
                plt.imshow(np.squeeze(x_batch_test_nopreprocess[img_ind])), plt.axis('off'), plt.title('original image')
                plt.show()

            
                fig, axs = plt.subplots(2, 2,figsize=(15,10))
                axs[0, 0].imshow(output_orig), axs[0, 0].axis('off'), axs[0, 0].set_title('original prediction')
                axs[0, 1].imshow(output_cf), axs[0, 1].axis('off'), axs[0, 1].set_title('modified prediction')
                axs[1, 0].plot(c_mean_fmap[0]),axs[1,0].set_ylim([0, np.max(c_mean_fmap)+1]), axs[1, 0].set_title('mean_fmaps')
                if args.counterfactual_PP:
                    axs[1, 1].plot(c_modified_mean_fmap_activations[0]),axs[1,1].set_ylim([0, np.max(c_mean_fmap)+1]), axs[1, 1].set_title('modified_mean_fmap_activations')
                else:
                    axs[1, 1].plot(PN_add[0],color='red'),axs[1,1].set_ylim([0, np.max(c_mean_fmap)+1]), axs[1, 1].set_title('PN_additions')
                    axs[1, 1].plot(c_mean_fmap[0]) ,axs[1,1].set_ylim([0, np.max(c_mean_fmap)+1]), plt.show()
                    
                plt.show()
                
                if not args.counterfactual_PP:
                    plt.plot(PN_add[0],color='red')
                    plt.plot(c_mean_fmap[0]), plt.title('PN_additions'),plt.ylim([0, np.max(c_mean_fmap)+1]), plt.show()
            #%% disabled PP prediction:
            enabled_filters = 1- t_fmatrix[0]
            dis_alter_probs, dis_fmaps, dis_mean_fmap, dis_modified_mean_fmap_activations,dis_alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),enabled_filters])#with eager
             
            print('\nDisabled PP prediction')
            print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',dis_alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')
            print( 'alter class: ',label_map[alter_class], '  prob: ',dis_alter_probs[0][alter_class].numpy()*100,'%')
            # print('alter_pre_softmax: ',alter_pre_softmax[0][0:10])
            if top_k_preds:
                 k=3
                 ind = np.argpartition(dis_alter_probs[0], -k)[-k:]
                 ind = ind[np.argsort(dis_alter_probs[0].numpy()[ind])]
                 
                 for i in range(k):
                     print('top ',str(i+1)+' predicted: ',label_map[ind[k-1-i]], ' with prob: ',dis_alter_probs[0][ind[k-1-i]].numpy()*100,'%')

            #%% analyze local PP disabling affect:
            analyze_local = False
            if analyze_local:
                #wrong prediction
                if np.argmax(alter_probs) != np.argmax(y_gt) and False:
                    print("wrong prediction")
                    local_misclassifications+=1
                else:
                    print("prediction unchanged")
                    pred_confidence_alter.append(alter_probs[0][np.argmax(alter_probs)].numpy()*100) 
                    pred_confidence_orig.append(pred_probs[0][np.argmax(pred_probs)].numpy()*100) 
                img_count+=1
                if img_count==28:
                    pass
                else:
                    continue
            #%% save figs for paper

            save_fig=False
            # save_fig = int(input("save fig?"))
            
        #      plt.savefig("mnist_0.png", dpi=None, facecolor='w', edgecolor='w',
        # orientation='portrait', papertype=None, format=None,
        # transparent=False, bbox_inches=None, pad_inches=0.1,
        # frameon=None, metadata=None)
            if save_fig:
                mName = args.model[:-1]+'_'+args.dataset
                plt.imshow(np.squeeze(x_batch_test_nopreprocess[img_ind])), plt.axis('off'),plt.savefig(fname="./figs_for_paper/"+mName+"_orig_alter_"+str(alter_class)+"_analysis_"+str(class_for_analysis)+"_"+str(actual_img_ind)+".png", dpi=None, bbox_inches = 'tight'), plt.show()                
                plt.imshow(output_orig), plt.axis('off'),plt.savefig(fname="./figs_for_paper/"+mName+"_output_orig_alter_"+str(alter_class)+"_analysis_"+str(class_for_analysis)+"_"+str(actual_img_ind)+"_"+str((pred_probs[0][np.argmax(y_gt)].numpy()*100).round(1))+".png", dpi=None, bbox_inches = 'tight'), plt.show()
                plt.imshow(output_cf), plt.axis('off'),plt.savefig(fname="./figs_for_paper/"+mName+"_output_cf_alter_"+str(alter_class)+"_analysis_"+str(class_for_analysis)+"_"+str(actual_img_ind)+"_"+str((alter_probs[0][alter_class].numpy()*100).round(1))+".png", dpi=None, bbox_inches = 'tight'), plt.show()
                plt.plot(c_mean_fmap[0]),plt.ylim([0, np.max(c_mean_fmap)+1]), plt.savefig(fname="./figs_for_paper/"+mName+"_mean_fmaps_alter_"+str(alter_class)+"_analysis_"+str(class_for_analysis)+"_"+str(actual_img_ind)+".png", dpi=None, bbox_inches = 'tight'), plt.show()
                plt.plot(c_modified_mean_fmap_activations[0]),plt.ylim([0, np.max(c_mean_fmap)+1]), plt.savefig(fname="./figs_for_paper/"+mName+"_modified_fmaps_alter_"+str(alter_class)+"_analysis_"+str(class_for_analysis)+"_"+str(actual_img_ind)+".png", dpi=None, bbox_inches = 'tight'), plt.show()
                
                np.save(file= "./figs_for_paper/"+mName+"_mean_fmaps_alter_"+str(alter_class)+"_analysis_"+str(class_for_analysis)+"_"+str(actual_img_ind),arr=c_mean_fmap[0])
                np.save(file= "./figs_for_paper/"+mName+"_modified_fmaps_alter_"+str(alter_class)+"_analysis_"+str(class_for_analysis)+"_"+str(actual_img_ind),arr=c_modified_mean_fmap_activations[0])


            #%% filter visualization on same iimage
            #sort by just magnitude or weighted magnitude?
            check_PN = False
            if check_PN:
               print("\nChecking PNs") 

            analyze_img_filters=True
            
            # analyze_img_filters = int(input("analyze filters on the input image?"))
            
            if analyze_img_filters:
                weighted_activation_magnitudes = c_modified_mean_fmap_activations[0]*W[:,alter_class] #w.r.t alter class
                # weighted_activation_magnitudes = W[:,alter_class] #w.r.t alter class
                sort_by_weighted_magnitude = True
               
                top_k=3
                if args.counterfactual_PP:
                    if sort_by_weighted_magnitude:
                        plt.plot(weighted_activation_magnitudes),plt.ylim([0, np.max(weighted_activation_magnitudes)+1]),plt.title('weigthed activations'), plt.show()
                        top_3_filters = np.argsort(weighted_activation_magnitudes)[-top_k:][::-1]#top 3
                    else:
                        top_3_filters = np.argsort(c_modified_mean_fmap_activations[0])[-top_k:][::-1]#top 3
                    #check_PN = True
                    if check_PN:
                        plt.plot(c_modified_mean_fmap_activations[0]),plt.ylim([0, np.max(c_mean_fmap)+1]), plt.show()

                        top_3_filters = np.array([ 15, 330, 158, 213, 456])#img 213 wrt top2 bronzed cowbird
#                        top_3_filters = np.array([ 15, 158, 297])#img 214
                        #top_3_filters = np.array([ 44, 274, 131, 193, 401])#img 240
                        # top_3_filters = np.array([  57, 401, 399,  44, 361])#img 240 PN wrt 9
                        #top_3_filters = np.array([  376,  57, 193, 359, 190])#img 652 PN wrt 9
                        # top_3_filters = np.array([ 330,  15, 456, 158, 469])#img 652 PN wrt 9
                        top_3_filters = np.array([376,  57, 193, 359, 190])#img 213                        

                        top_3_filters = top_3_filters[:top_k]
                else:
                    top_3_filters = np.argsort(PN_add[0])[-5:][::-1]#top 3 filters to which largest addition is made
                PP_filter_list = np.where(filters_off[0]>0)#same for PP or PN

                filter_visualization_same_image(model,combined, x_batch_test[img_ind],actual_img_ind,top_3_filters, args,show_images=False, gradCAM=True, RF = True, combined_heatmaps = False)
                
            #sys.exit()
           
            #%% filter visualization on class specific or all images
            #sort by just magnitude or weighted magnitude?
            analyze_filters=False
            # analyze_filters = int(input("analyze image filters?"))
            
            if analyze_filters:
                weighted_activation_magnitudes = c_modified_mean_fmap_activations[0]*W[:,alter_class] #w.r.t alter class
                # weighted_activation_magnitudes = W[:,alter_class] #w.r.t alter class
                sort_by_weighted_magnitude = True
               
                top_k=3
                if args.counterfactual_PP:
                    if sort_by_weighted_magnitude:
                        plt.plot(weighted_activation_magnitudes),plt.ylim([0, np.max(weighted_activation_magnitudes)+1]),plt.title('weigthed activations'), plt.show()
                        top_3_filters = np.argsort(weighted_activation_magnitudes)[-top_k:][::-1]#top 3
                    else:
                        top_3_filters = np.argsort(c_modified_mean_fmap_activations[0])[-top_k:][::-1]#top 3
                    if check_PN:
                        plt.plot(c_modified_mean_fmap_activations[0]),plt.ylim([0, np.max(c_mean_fmap)+1]), plt.show()

                        #top_3_filters = np.array([ 15, 330, 158, 213, 456])
                        #top_3_filters = np.array([ 44, 193, 376])#img 213                        
                        # top_3_filters = np.array([ 15, 158, 297])#img 214                        
                        # top_3_filters = np.array([ 44, 274, 131, 193, 401])#img 240
                        #top_3_filters = np.array([  57, 401, 399,  44, 361])#img 240 PN wrt 9
                        # top_3_filters = np.array([  376,  57, 193, 359, 190])#img 652 PN wrt 9
                        # top_3_filters = np.array([ 330,  15, 456, 158, 469])#img 652 PN wrt 9
                        top_3_filters = np.array([376,  57, 193, 359, 190])#img 213                        

                        
                else:
                    top_3_filters = np.argsort(PN_add[0])[-5:][::-1]#top 3 filters to which largest addition is made
                
                global_filters = False
                if global_filters:
                    print("Class specific global filters analysis")
                    mName = args.model[:-1]+'_'+args.dataset
                    # filter_magnitude_cf_alter_class = np.load(file= "./figs_for_paper/misclassification case/"+mName+"_normalized_filter_magnitude_cf_"+str(alter_class)+"_train_set.np.npy")
                    # filter_histogram_cf_alter_class = np.load(file= "./figs_for_paper/misclassification case/"+mName+"_filter_histogram_cf_"+str(alter_class)+"_train_set.np.npy")
                    filter_magnitude_cf_alter_class = np.load(file= "./figs_for_paper/"+mName+"_normalized_filter_magnitude_cf_"+str(alter_class)+"_train_set.np.npy")
                    filter_histogram_cf_alter_class = np.load(file= "./figs_for_paper/"+mName+"_filter_histogram_cf_"+str(alter_class)+"_train_set.np.npy")
                    
                    normalized_filter_magnitude_cf_alter_class = filter_magnitude_cf_alter_class/max(filter_histogram_cf_alter_class)
                    
                    plt.plot(normalized_filter_magnitude_cf_alter_class), plt.ylim([0, np.max(normalized_filter_magnitude_cf_alter_class)+1]), plt.title("average_filter_magnitude_cf_"+str(alter_class)), plt.show()
                    
                    top_3_filters = np.argsort(filter_histogram_cf_alter_class)[-top_k:][::-1]#top 3
                    
                    
                PP_filter_list = np.where(filters_off[0]>0)#same for PP or PN
                k=5 #top k images
                for i in range(top_k):
                    filter_visualization_top_k(model,test_gen,top_3_filters[i],k,args,show_images=False, gradCAM=True, class_specific_top_k=True, RF = True)
                
                print("\n")
                for i in range(top_k): print("filter "+str(top_3_filters[i])+" weight for class "+str(args.alter_class)+": "+str(W[top_3_filters[i],args.alter_class].numpy()))

            #sys.exit()
#%%
            select_analyze_filters=False
            # select_analyze_filters = int(input("analyze some selected filters?"))
            
            if select_analyze_filters:
                selected_filter = [int(input("choose filter:"))]

               
                top_3_filters = selected_filter
                PP_filter_list = np.where(filters_off[0]>0)#s0ame for PP or PN
                k=5 #top k images
                for i in range(1):
                    filter_visualization_top_k(model,test_gen,top_3_filters[i],k,args,show_images=False, gradCAM=True, class_specific_top_k=False, RF = True)
                
                print("\n")
                for i in range(1): print("filter "+str(top_3_filters[i])+" weight for class "+str(args.alter_class)+": "+str(W[top_3_filters[i],args.alter_class].numpy()))
            
            #%% disable individual filter only
            disable_one_filter=False
            # disable_one_filter = int(input("disable chosen filters individually?"))
            
            if disable_one_filter:
                #one_filter = int(input("choose filter to disbale:"))
                top_10_fils = np.argsort(c_modified_mean_fmap_activations[0])[-10:][::-1]
                for fil in range (len(top_10_fils)):
                    enabled_filters = np.ones(512)
                    
                    enabled_filters[top_10_fils[fil]]=0.0
                    enabled_filters = tf.convert_to_tensor(enabled_filters)
                    
                    
                    dis_one_alter_probs, dis_one_fmaps, dis_one_mean_fmap, dis_one_modified_mean_fmap_activations,dis_one_alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),enabled_filters])#with eager
                     
                    print('\nDisabled filter ',top_10_fils[fil],' prediction')
                    print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',dis_one_alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')
                    print( 'alter class: ',label_map[alter_class], '  prob: ',dis_one_alter_probs[0][alter_class].numpy()*100,'%')
                    # print('alter_pre_softmax: ',alter_pre_softmax[0][0:10])
                    if top_k_preds:
                         k=3
                         ind = np.argpartition(dis_one_alter_probs[0], -k)[-k:]
                         ind = ind[np.argsort(dis_one_alter_probs[0].numpy()[ind])]
                         
                         for i in range(k):
                             print('top ',str(i+1)+' predicted: ',label_map[ind[k-1-i]], ' with prob: ',dis_one_alter_probs[0][ind[k-1-i]].numpy()*100,'%')
            
            
            #%% model accuracy on disabled filters 
            find_model_accuracy=False
            # find_model_accuracy = int(input("find model accuracy on disabled filters?"))
            
            if find_model_accuracy:
                enabled_filters = np.ones(512)
                for i in tf.where(filters_off[0]>0):
                    enabled_filters[tuple(i)]=0.0
                enabled_filters = tf.convert_to_tensor(enabled_filters)                
                
                mName = args.model[:-1]+'_'+args.dataset
                # filter_magnitude_cf_alter_class = np.load(file= "./figs_for_paper/misclassification case/"+mName+"_normalized_filter_magnitude_cf_"+str(alter_class)+"_train_set.np.npy")
                # filter_histogram_cf_alter_class = np.load(file= "./figs_for_paper/misclassification case/"+mName+"_filter_histogram_cf_"+str(alter_class)+"_train_set.np.npy")
                filter_magnitude_cf_alter_class = np.load(file= "./figs_for_paper/"+mName+"_normalized_filter_magnitude_cf_"+str(alter_class)+"_train_set.np.npy")
                filter_histogram_cf_alter_class = np.load(file= "./figs_for_paper/"+mName+"_filter_histogram_cf_"+str(alter_class)+"_train_set.np.npy")
                
                normalized_filter_magnitude_cf_alter_class = filter_magnitude_cf_alter_class/max(filter_histogram_cf_alter_class)

                plt.plot(filter_histogram_cf_alter_class), plt.ylim([0, np.max(filter_histogram_cf_alter_class)+1]),plt.title("activated_filters_count_cf_"+str(alter_class)),  plt.show()
                plt.plot(normalized_filter_magnitude_cf_alter_class), plt.ylim([0, np.max(normalized_filter_magnitude_cf_alter_class)+1]), plt.title("average_filter_magnitude_cf_"+str(alter_class)), plt.show()
                
                global_filters_alter_class = np.zeros_like(normalized_filter_magnitude_cf_alter_class) 
                random_filters_alter_class = np.zeros_like(normalized_filter_magnitude_cf_alter_class) 
                for i in tf.where(filter_histogram_cf_alter_class>0):
                 	global_filters_alter_class[tuple(i)]=1.0
                
                enabled_filters = 1-global_filters_alter_class
                print("\n Global PP filters disabled: ", np.sum(global_filters_alter_class))
                
                #random test: randomly disable some number of filters and see change in performance
                while np.sum(random_filters_alter_class)!=40:
                    rndIndx = np.random.randint(512)
                    random_filters_alter_class[rndIndx]=1.0          
                random_enabled_filters = 1-random_filters_alter_class
                # print("randoms filters disabled:", np.sum(random_filters_alter_class))          

                # filter_magnitude_cf_alter_class = np.load(file= "./figs_for_paper/misclassification case/"+mName+"_normalized_filter_magnitude_cf_"+str(9)+"_train_set.np.npy")
                # normalized_filter_magnitude_cf_alter_class = filter_magnitude_cf_alter_class#/max(filter_histogram_cf_alter_class)
                
                # global_filters_alter_class2 = np.copy(normalized_filter_magnitude_cf_alter_class) 
                # for i in tf.where(normalized_filter_magnitude_cf_alter_class>0):
                # 	global_filters_alter_class2[tuple(i)]=1.0
                
                # filter_magnitude_cf_alter_class = np.load(file= "./figs_for_paper/misclassification case/"+mName+"_normalized_filter_magnitude_cf_"+str(108)+"_train_set.np.npy")
                # normalized_filter_magnitude_cf_alter_class = filter_magnitude_cf_alter_class#/max(filter_histogram_cf_alter_class)
                
                # global_filters_alter_class3 = np.copy(normalized_filter_magnitude_cf_alter_class) 
                # for i in tf.where(normalized_filter_magnitude_cf_alter_class>0):
                # 	global_filters_alter_class3[tuple(i)]=1.0
                
                    
                # combined = global_filters_alter_class+global_filters_alter_class2 + global_filters_alter_class3
                
                # enabled_filters = np.ones(512)
                # for i in tf.where(combined>0):
                #     enabled_filters[tuple(i)]=0.0
                # enabled_filters = tf.convert_to_tensor(enabled_filters)
                
                #enabled_filters = np.ones(512)
                test_acc, test_loss = model_accuracy_filters(model,test_gen, enabled_filters, args)
                

            
#%% misclassification analysis
            
            global_analysis = False
            if misclassification_analysis and False:
                mName = args.model[:-1]
                
                if mName != 'myCNN':
                    ##### for model of alter class 9
                    filter_histogram_cf_alter_class = np.load(file= "./figs_for_paper/misclassification case/"+mName+"_filter_histogram_cf_"+str(alter_class)+"_train_set.np.npy")
                    filter_magnitude_cf_alter_class = np.load(file= "./figs_for_paper/misclassification case/"+mName+"_normalized_filter_magnitude_cf_"+str(alter_class)+"_train_set.np.npy")
                    
                    normalized_filter_magnitude_cf_alter_class = filter_magnitude_cf_alter_class/max(filter_histogram_cf_alter_class)
                    plt.plot(filter_histogram_cf_alter_class), plt.ylim([0, np.max(filter_histogram_cf_alter_class)+1]),plt.title("activated_filters_count_cf_"+str(alter_class)),  plt.show()
                    plt.plot(normalized_filter_magnitude_cf_alter_class), plt.ylim([0, np.max(normalized_filter_magnitude_cf_alter_class)+1]), plt.title("average_filter_magnitude_cf_"+str(alter_class)), plt.show()
                    
                c_modified_mean_fmap_activations_alter_class = c_modified_mean_fmap_activations
                    # plt.plot(c_modified_mean_fmap_activations_9[0]),plt.ylim([0, np.max(c_mean_fmap)+1]), plt.title("CFE_9_216"), plt.show()
                    
                ##### for model of alter class 2
                combined.load_weights(filepath=weights_path+'/counterfactual_combined_model_fixed_'+str(label_map[args.alter_class_2])+'_alter_class.hdf5')
                alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(np.expand_dims(x_batch_test[img_ind],0))

                if mName != 'myCNN':
                    filter_histogram_cf_class_2 = np.load(file= "./figs_for_paper/misclassification case/"+mName+"_filter_histogram_cf_"+str(args.alter_class_2)+"_train_set.np.npy")
                    filter_magnitude_cf_class_2 = np.load(file= "./figs_for_paper/misclassification case/"+mName+"_normalized_filter_magnitude_cf_"+str(args.alter_class_2)+"_train_set.np.npy")

                    normalized_filter_magnitude_cf_class_2 = filter_magnitude_cf_class_2/max(filter_histogram_cf_class_2)
                    plt.plot(filter_histogram_cf_class_2), plt.ylim([0, np.max(filter_histogram_cf_class_2)+1]),  plt.title("activated_filters_count_cf_"+str(args.alter_class_2)), plt.show()
                    plt.plot(normalized_filter_magnitude_cf_class_2), plt.ylim([0, np.max(normalized_filter_magnitude_cf_class_2)+1]), plt.title("average_filter_magnitude_cf_"+str(args.alter_class_2)), plt.show()

                c_modified_mean_fmap_activations_class_2 = modified_mean_fmap_activations
                # plt.plot(c_modified_mean_fmap_activations_170),plt.ylim([0, np.max(c_mean_fmap)+1]), plt.title("CFE_170_216"), plt.show()
                
                
                
                ########################  
                #check if dis-common filters between these two are changing the prediction
                filters_alter_class = np.copy(c_modified_mean_fmap_activations_alter_class) 
                for i in tf.where(c_modified_mean_fmap_activations_alter_class>0):
                    filters_alter_class[tuple(i)]=1.0
                
                filters_class_2 = np.copy(c_modified_mean_fmap_activations_class_2)
                for i in tf.where(c_modified_mean_fmap_activations_class_2>0):
                    filters_class_2[tuple(i)]=1.0
                    
                common = filters_alter_class * filters_class_2
                
                dis_common_alter_class = (1-common)*filters_alter_class
                dis_common_class_2 = (1-common)*filters_class_2


                ########################                
                #check common filters/agreement between CFE model local and global filters
                filters_alter_class = np.copy(c_modified_mean_fmap_activations_alter_class) 
                for i in tf.where(c_modified_mean_fmap_activations_alter_class>0):
                    filters_alter_class[tuple(i)]=1.0
                
                if global_analysis:
                    global_filters_alter_class = np.copy(normalized_filter_magnitude_cf_alter_class) 
                    for i in tf.where(normalized_filter_magnitude_cf_alter_class>0):
                        global_filters_alter_class[tuple(i)]=1.0
                        
                    global_common_alter_class = filters_alter_class * global_filters_alter_class
                

                filters_class_2 = np.copy(c_modified_mean_fmap_activations_class_2) 
                for i in tf.where(c_modified_mean_fmap_activations_class_2>0):
                    filters_class_2[tuple(i)]=1.0
                
                if global_analysis:
                    global_filters_class_2 = np.copy(normalized_filter_magnitude_cf_class_2) 
                    for i in tf.where(normalized_filter_magnitude_cf_class_2>0):
                        global_filters_class_2[tuple(i)]=1.0
                        
                    global_common_class_2 = filters_class_2 * global_filters_class_2
                #########################
                
                #switch off each filter one by one
                
                my_fmatrix = np.copy(c_modified_mean_fmap_activations_class_2)
                for i in tf.where(my_fmatrix>0):
                    my_fmatrix[tuple(i)]=1.0
                
                #deactivate each filter
                if False:
                    for i in tf.where(my_fmatrix>0):
                        filters_off = np.copy(my_fmatrix)
                        filters_off[tuple(i)]=0.0
                        alter_probs, c_fmaps, c_mean_fmap, c_modified_mean_fmap_activations,alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),filters_off])#with eager
                        
                        print('\nfilter modification analysis')
                        print( 'gt class: ',label_map[np.argmax(alter_probs)], '  prob: ',alter_probs[0][np.argmax(alter_probs)].numpy()*100,'%')
                        print( 'alter class: ',label_map[alter_class], '  prob: ',alter_probs[0][alter_class].numpy()*100,'%')
                        #print('alter_pre_softmax: ',alter_pre_softmax[0][0:10])
        
                        if top_k_preds:
                            k=3
                            ind = np.argpartition(alter_probs[0], -k)[-k:]
                            ind = ind[np.argsort(alter_probs[0].numpy()[ind])]
                            
                            for i in range(k):
                                print('top ',str(i+1)+' predicted: ',label_map[ind[k-1-i]], ' with prob: ',alter_probs[0][ind[k-1-i]].numpy()*100,'%')

                        plt.plot(c_modified_mean_fmap_activations[0]), plt.ylim([0,np.max(c_mean_fmap)+1] ), plt.title('modified_mean_fmap_activations'), plt.show()
               
                #####################                
                #For misclassification analysis:
                #    max_incorrect = Actual-minimum correct
                #    max_correct = Actual-minimum in-correct
                
                actual = np.copy(c_mean_fmap[0])
                minimum_correct = np.copy(c_modified_mean_fmap_activations_alter_class[0])
                minimum_incorrect = np.copy(c_modified_mean_fmap_activations_class_2[0])
                
                for i in tf.where(actual>0):
                    actual[tuple(i)]=1.0
                for i in tf.where(minimum_correct>0):
                    minimum_correct[tuple(i)]=1.0
                for i in tf.where(minimum_incorrect>0):
                    minimum_incorrect[tuple(i)]=1.0
                
                
                if global_analysis:
                    analysis_level="global"
                    print("\n\nanalysing using global PP/PNs")
                    max_incorrect = actual - global_filters_alter_class#global_filters_9#minimum_correct#global_filters_9#minimum_correct
                    max_correct   = actual - global_filters_class_2#global_filters_170#dis_common_170#minimum_incorrect
                else:
                    analysis_level="local"                    
                    print("\n\nanalysing using local PP/PNs")
                    max_incorrect = actual - minimum_correct#global_filters_9#minimum_correct#global_filters_9#minimum_correct
                    max_correct   = actual - minimum_incorrect#global_filters_170#dis_common_170#minimum_incorrect
                
                if global_analysis:
                    #check common and dis-common PP filters of the two classes                    
                    common_global = global_filters_alter_class * global_filters_class_2
                    discommon_alter_class = (1-common_global)*global_filters_alter_class                
                    discommon_class_2 = (1-common_global)*global_filters_class_2                
                
                ########################
                
                my_fmatrix = max_incorrect#actual - discommon_alter_class#max_incorrect#max_incorrect#1-common_global#max_incorrect#max_incorrect#common_global + discommon_alter_class#max_incorrect
                
                ######################
                alter_probs, c_fmaps, c_mean_fmap, c_modified_mean_fmap_activations,alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),my_fmatrix])#with eager
                
                print('\nfilter modification analysis: max_incorrect ('+analysis_level+')')
                print( 'gt class: ',label_map[np.argmax(alter_probs)], '  prob: ',alter_probs[0][np.argmax(alter_probs)].numpy()*100,'%')
                print( 'alter class: ',label_map[alter_class], '  prob: ',alter_probs[0][alter_class].numpy()*100,'%')
                #print('alter_pre_softmax: ',alter_pre_softmax[0][0:10])

                if top_k_preds:
                    k=3
                    ind = np.argpartition(alter_probs[0], -k)[-k:]
                    ind = ind[np.argsort(alter_probs[0].numpy()[ind])]
                    
                    for i in range(k):
                        print('top ',str(i+1)+' predicted: ',label_map[ind[k-1-i]], ' with prob: ',alter_probs[0][ind[k-1-i]].numpy()*100,'%')

                plt.plot(c_modified_mean_fmap_activations[0]), plt.ylim([0,np.max(c_mean_fmap)+1] ), plt.title('modified_mean_fmap_activations'), plt.show()
                
                
                #todo: highlight the plotted "actual - global_filters_9" filters that are removed
                if global_analysis:
                    filters_removed = discommon_alter_class#global_filters_alter_class#common_global#global_filters_alter_class#common_global#global_filters_alter_class
                else:
                    filters_removed = minimum_correct
                removed_filters = mean_fmaps[0]*(actual - (1-filters_removed))
                plt.plot(c_modified_mean_fmap_activations[0],color=None), plt.ylim([0, np.max(mean_fmaps)+1]), plt.title('highlighted mean_fmaps'),
                plt.plot(removed_filters,color='red'), plt.show()

                if global_analysis:
                    #todo: which PPs were/were-not activated?
                    global_activated = normalized_filter_magnitude_cf_alter_class*(actual*filters_removed)
                    plt.plot(global_activated,color=None), plt.ylim([0, np.max(normalized_filter_magnitude_cf_alter_class)+1]), plt.title('global activated filters '+str(alter_class)),                
                    global_nonactivated = normalized_filter_magnitude_cf_alter_class*((1-actual)*filters_removed)
                    plt.plot(global_nonactivated,color='red'), plt.ylim([0, np.max(normalized_filter_magnitude_cf_alter_class)+1]), plt.show(),
    
                    
                    output_cf,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,alter_class,image_nopreprocessed=np.expand_dims(x_batch_test_nopreprocess[img_ind],0),fmatrix=t_fmatrix,image_weight=0.7)#np.argmin(y_batch_test[img_ind])
                    
                    plt.imshow(output_cf), plt.axis('off'), plt.title('thresh prediction')
                    plt.show()
                    # plt.imshow(np.squeeze(x_batch_test_nopreprocess[img_ind])), plt.axis('off'), plt.title('original image')
                    # plt.show()
                    
                    ##############
                    #plot global common filters
                    global_activated = normalized_filter_magnitude_cf_alter_class*(1-common_global)
                    plt.plot(global_activated,color=None), plt.ylim([0, np.max(normalized_filter_magnitude_cf_alter_class)+1]), plt.title('global common filters '+str(alter_class)),                
                    global_nonactivated = normalized_filter_magnitude_cf_alter_class*(common_global)
                    plt.plot(global_nonactivated,color='red'), plt.ylim([0, np.max(normalized_filter_magnitude_cf_alter_class)+1]), plt.show(),
    
                    global_activated = normalized_filter_magnitude_cf_class_2*(1-common_global)
                    plt.plot(global_activated,color=None), plt.ylim([0, np.max(normalized_filter_magnitude_cf_class_2)+1]), plt.title('global common filters '+str(args.alter_class_2)),                
                    global_nonactivated = normalized_filter_magnitude_cf_class_2*(common_global)
                    plt.plot(global_nonactivated,color='red'), plt.ylim([0, np.max(normalized_filter_magnitude_cf_class_2)+1]), plt.show(),
                
                
                ##############
                #check global filters weights positive or negative
                weights_global_alter_class = weights_alter_class*minimum_correct#global_filters_alter_class
                plt.plot(weights_global_alter_class),plt.show()
                
                weights_global_class_2 = weights_class_2*minimum_correct#global_filters_alter_class#global_filters_class_2
                plt.plot(weights_global_class_2),plt.show()

                
                ########################
                #2nd alter class
                second_example=False
                if second_example:
                    combined.load_weights(filepath=weights_path+'/counterfactual_combined_model_fixed_'+str(label_map[args.alter_class_2])+'_alter_class.hdf5')
                    alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(np.expand_dims(x_batch_test[img_ind],0))
                    filters_off = fmatrix
                    t_fmatrix = filters_off.numpy()
                    for i in tf.where(fmatrix>0):
                        t_fmatrix[tuple(i)]=1.0
                    t_fmatrix = tf.convert_to_tensor(t_fmatrix)
                    alter_probs, c_fmaps, c_mean_fmap, c_modified_mean_fmap_activations,alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),t_fmatrix])#with eager
                    
                    print('\n\n 2nd thresholded counterfactual')
                    print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')
                    print( 'alter class: ',label_map[args.alter_class_2], '  prob: ',alter_probs[0][args.alter_class_2].numpy()*100,'%')
                   # print('alter_pre_softmax: ',alter_pre_softmax[0][0:10])
                    if top_k_preds:
                         k=3
                         ind = np.argpartition(alter_probs[0], -k)[-k:]
                         ind = ind[np.argsort(alter_probs[0].numpy()[ind])]
                         
                         for i in range(k):
                             print('top ',str(i+1)+' predicted: ',label_map[ind[k-1-i]], ' with prob: ',alter_probs[0][ind[k-1-i]].numpy()*100,'%')

                    #modified_model, modified_fmaps = check_histogram_top_filter_result(model,filters_off,x_batch_test[img_ind],y_gt,label_map,args)
                    #plt.plot(fmatrix[0]), plt.title('fmatrix'), plt.show()
    
                    #modified_mean_fmap_activations = GlobalAveragePooling2D()(modified_fmaps)
                    plt.plot(c_modified_mean_fmap_activations[0]), plt.title('thresh_mean_fmap_activations'), plt.show()
                    
                    output_cf,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,170,image_nopreprocessed=np.expand_dims(x_batch_test_nopreprocess[img_ind],0),fmatrix=t_fmatrix,image_weight=0.7)#np.argmin(y_batch_test[img_ind])
                    
                    plt.imshow(output_cf), plt.axis('off'), plt.title('thresh prediction')
                    plt.show()
                    # plt.imshow(np.squeeze(x_batch_test_nopreprocess[img_ind])), plt.axis('off'), plt.title('original image')
                    # plt.show()
    
                
                    fig, axs = plt.subplots(2, 2,figsize=(15,10))
                    axs[0, 0].imshow(output_orig), axs[0, 0].axis('off'), axs[0, 0].set_title('original prediction')
                    axs[0, 1].imshow(output_cf), axs[0, 1].axis('off'), axs[0, 1].set_title('modified prediction')
                    axs[1, 0].plot(c_mean_fmap[0]),axs[1,0].set_ylim([0, np.max(c_mean_fmap)+1]), axs[1, 0].set_title('mean_fmaps')
                    axs[1, 1].plot(c_modified_mean_fmap_activations[0]),axs[1,1].set_ylim([0, np.max(c_mean_fmap)+1]), axs[1, 1].set_title('modified_mean_fmap_activations')
                    plt.show()
                    
                    filters_class_2 = np.copy(c_modified_mean_fmap_activations) 
                    for i in tf.where(c_modified_mean_fmap_activations>0):
                        filters_class_2[tuple(i)]=1.0                
                        
                    global_common_class_2 = filters_class_2 * global_filters_class_2
                    
                    
                     #####################                
                    #For misclassification analysis:
                    #    max_incorrect = Actual-minimum correct
                    #    max_correct = Actual-minimum in-correct
                    
                    actual = np.copy(c_mean_fmap[0])
                    minimum_correct = np.copy(c_modified_mean_fmap_activations_alter_class[0])
                    minimum_incorrect = np.copy(c_modified_mean_fmap_activations)
                    
                    for i in tf.where(actual>0):
                        actual[tuple(i)]=1.0
                    for i in tf.where(minimum_correct>0):
                        minimum_correct[tuple(i)]=1.0
                    for i in tf.where(minimum_incorrect>0):
                        minimum_incorrect[tuple(i)]=1.0
                    
                    max_incorrect = actual - global_filters_class_2#global_filters_9#minimum_correct
                    max_correct   = actual - global_filters_class_2#minimum_incorrect#global_filters_170#global_filters_170#global_filters_170#dis_common_170#minimum_incorrect
    
    
    
                    if global_analysis:
                        analysis_level="global"
                        print("analysing using global PP/PNs")
                        max_incorrect = actual - global_filters_alter_class#global_filters_9#minimum_correct#global_filters_9#minimum_correct
                        max_correct   = actual - global_filters_class_2#global_filters_170#dis_common_170#minimum_incorrect
                    else:
                        analysis_level="local"                    
                        print("analysing using local PP/PNs")
                        max_incorrect = actual - minimum_correct#global_filters_9#minimum_correct#global_filters_9#minimum_correct
                        max_correct   = actual - minimum_incorrect#global_filters_170#dis_common_170#minimum_incorrect
        
                    ########################
                    
                    my_fmatrix = max_correct
                    
                    ######################
                    alter_probs, c_fmaps, c_mean_fmap, c_modified_mean_fmap_activations,alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),my_fmatrix])#with eager
                    
                    print('\nfilter modification analysis: max correct ('+analysis_level+')')
                    print( 'gt class: ',label_map[np.argmax(alter_probs)], '  prob: ',alter_probs[0][np.argmax(alter_probs)].numpy()*100,'%')
                    print( 'alter class: ',label_map[alter_class], '  prob: ',alter_probs[0][alter_class].numpy()*100,'%')
                    #print('alter_pre_softmax: ',alter_pre_softmax[0][0:10])

                    if top_k_preds:
                         k=3
                         ind = np.argpartition(alter_probs[0], -k)[-k:]
                         ind = ind[np.argsort(alter_probs[0].numpy()[ind])]
                         
                         for i in range(k):
                             print('top ',str(i+1)+' predicted: ',label_map[ind[k-1-i]], ' with prob: ',alter_probs[0][ind[k-1-i]].numpy()*100,'%')
    
    
                    plt.plot(c_modified_mean_fmap_activations[0]), plt.ylim([0,np.max(c_mean_fmap)+1] ), plt.title('modified_mean_fmap_activations'), plt.show()
                    
                    
                    #todo: highlight the plotted actual - global_filters_9 filters that are removed
                    if global_analysis:
                        filters_removed = global_filters_class_2
                    else:
                        filters_removed = minimum_incorrect
                    
                    removed_filters = mean_fmaps[0]*(actual - (1-filters_removed))
                    plt.plot(c_modified_mean_fmap_activations[0],color=None), plt.ylim([0, np.max(mean_fmaps)+1]), plt.title('highlighted mean_fmaps'),
                    plt.plot(removed_filters,color='red'), plt.show()
                    
                    #PPs not activated
                    global_activated = normalized_filter_magnitude_cf_class_2*(actual*filters_removed)
                    plt.plot(global_activated,color=None), plt.ylim([0, np.max(normalized_filter_magnitude_cf_class_2)+1]), plt.title('global activated filters '+str(args.alter_class_2)),                
                    global_nonactivated = normalized_filter_magnitude_cf_class_2*((1-actual)*filters_removed)
                    plt.plot(global_nonactivated,color='red'), plt.ylim([0, np.max(normalized_filter_magnitude_cf_class_2)+1]), plt.show(),


                    output_cf,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,alter_class,image_nopreprocessed=np.expand_dims(x_batch_test_nopreprocess[img_ind],0),fmatrix=t_fmatrix,image_weight=0.7)#np.argmin(y_batch_test[img_ind])
                    
                    plt.imshow(output_cf), plt.axis('off'), plt.title('thresh prediction')
                    plt.show()
                    # plt.imshow(np.squeeze(x_batch_test_nopreprocess[img_ind])), plt.axis('off'), plt.title('original image')
                    # plt.show()
            #%% perturb the input image according to complement of heatmap
            perturb=False
            if perturb:
                img = x_batch_test_nopreprocess[img_ind]
                
                output,only_heatmap = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,alter_class,image_nopreprocessed=np.expand_dims(x_batch_test_nopreprocess[img_ind],0),fmatrix=fmatrix,image_weight=0.0)#np.argmin(y_batch_test[img_ind])
                
                masked_preprocessed = get_heatmap_only(only_heatmap,x_batch_test[img_ind])
                masked_non_preprocessed = get_heatmap_only(only_heatmap,img)
    
                plt.imshow(masked_non_preprocessed), plt.axis('off'), plt.title('perturbed'),plt.show()
                
                perturbed_probs, perturbed_fmaps, perturbed_mean_fmap, perturbed_modified_mean_fmap_activations,perturbed_pre_softmax = model([np.expand_dims(masked_preprocessed,0),default_fmatrix])#with eager
                    
                print('\nperturbed')
                #print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',perturbed_probs[0][np.argmax(y_gt)].numpy()*100,'%')
                print( 'predicted class: ',label_map[np.argmax(perturbed_probs)], '  prob: ',perturbed_probs[0][np.argmax(perturbed_probs)].numpy()*100,'%')
                print( 'alter class: ',label_map[alter_class], '  prob: ',perturbed_probs[0][alter_class].numpy()*100,'%')
                #print('perturbed_pre_softmax: ',perturbed_pre_softmax[0][0:10])
                
                output,_ = explainer.explain((np.expand_dims(masked_preprocessed,0),None),model,np.argmax(perturbed_probs),image_nopreprocessed=np.expand_dims(masked_non_preprocessed,0),fmatrix=default_fmatrix,image_weight=0.7)#np.argmin(y_batch_test[img_ind])
                
                plt.plot(perturbed_modified_mean_fmap_activations[0]), plt.title('perturbed_modified_mean_fmap_activations'), plt.show()
        
                plt.imshow(output), plt.axis('off'), plt.title('perturbed prediction')
                plt.show()
                print(' ')
            #break
        #break
sys.exit()
#%% check indivual filter gradCAM?
img_ind=2
alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,pre_softmax = combined(np.expand_dims(x_batch_test[img_ind],0))

fmatrix = np.zeros_like(fmatrix)
fmatrix[:,355] =1#286, 355, 41

pred_probs,fmaps,mean_fmaps,modified_mean_fmap_activations,pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),fmatrix], training=False)#with eager

important_filters_1 = np.where(mean_fmaps[0,:]>=30)
important_filters_2 = np.where(modified_mean_fmap_activations[0,:]>=15)

#pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%')

plt.plot(mean_fmaps[0]), plt.title('mean_fmaps'),plt.show()
plt.plot(modified_mean_fmap_activations[0]), plt.title('modified_mean_fmap_activations'),plt.show()

output = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,np.argmax(y_batch_test[img_ind]),image_nopreprocessed=np.expand_dims(x_batch_test_nopreprocess[img_ind],0),fmatrix=fmatrix)

plt.imshow(output), plt.axis('off'), plt.title('modified prediction')
plt.show()
plt.imshow(x_batch_test_nopreprocess[img_ind]), plt.axis('off'), plt.title('original image')
plt.show()

#%% check product sum idea
#pred_probs,fmaps,mean_fmaps,modified_mean_fmap_activations = model([np.expand_dims(x_batch_test[img_ind],0),fmatrix], training=False)#with eager

W = model.weights[-2]
b = model.weights[-1]
modified_mean_fmap_activations

# match this score --> numpy=array([[0.25339222, 0.7466078 ]] for cat img_ind =0

#fmap shape = 1x512
# W shape   = 512 x 2
# b shape   = 2

prod_sum1 = np.sum(tf.multiply(modified_mean_fmap_activations,W[:,0])) + b[0]
prod_sum2 = np.sum(tf.multiply(modified_mean_fmap_activations,W[:,1])) + b[1]

print('prod_sum1:',prod_sum1)
print('prod_sum2:', prod_sum2)

softmx = tf.keras.activations.softmax(tf.convert_to_tensor(np.reshape([prod_sum1,prod_sum2],(1,2))))
print('my softmax',softmx)
#%% full test set result
model = modified_model
batches=math.ceil(test_gen.n/test_gen.batch_size)
test_gen.reset()
#for step,(x_batch_train, y_batch_train) in enumerate(dataset):
with tqdm(total=batches, file=sys.stdout) as progBar:
    for step in range(batches):
      x_batch_test, y_batch_test = next(test_gen)
      
      probs = test_step(x_batch_test, y_batch_test)
      
      progBar.set_postfix(loss=test_loss_metric.result().numpy(), acc=test_acc_metric.result().numpy())
      progBar.update()
        
  
     # Display metrics at the end of each epoch.
    test_acc = test_acc_metric.result()
    test_loss = test_loss_metric.result()
print('\nTest loss:', test_loss.numpy())
print('Test accuracy:', test_acc.numpy())
 
#%%
x_batch_test,y_batch_test = next(test_gen)
explainer = GradCAM()


#%% check filter importance
if (args.test_filter_importance and not args.save_filter_importance):
    img_ind=14
    model.load_weights(filepath=weights_path+'/model.hdf5')
    
    if args.full_standard:
        pred_probs = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
    else:
        pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager


    y_gt = y_batch_test[img_ind]
    
    plt.figure(figsize = (20,2))
    plt.imshow(x_batch_test[img_ind].squeeze(),cmap='gray')
    plt.axis('off')
    plt.show()
    print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
    print ('actual: ', label_map[np.argmax(y_gt)])
    
    original_prob=np.max(pred_probs)
    test_filter_importance(model,weights_path,x_batch_test[img_ind],y_batch_test[img_ind],label_map,original_prob,img_ind)
    
    ##%% in-code method to modify filters 
    test_filter_importance_in_code_method(model,weights_path,x_batch_test[img_ind],y_batch_test[img_ind],label_map,original_prob,img_ind)
      

#%% save filter importance
for i in range(len(model.weights)):
      print(len(model.weights)-i,  model.weights[i].shape)
#%% test in batch
if (args.save_filter_importance and True):
    #class_for_analysis = 0
    #print ('class for analysis: ', label_map[class_for_analysis])
    trainSplit = False# train or test split
    
    
    if trainSplit:
        batches=math.ceil(train_gen.n/train_gen.batch_size)
    
        train_gen.reset() #resets batch index to 0
        gen=train_gen
    else:
        batches=math.ceil(test_gen.n/test_gen.batch_size)
    
        test_gen.reset() #resets batch index to 0
        gen=test_gen
    

    img_count=0
    for k in range(batches):
        
        x_batch_test,y_batch_test = next(gen)
        
        if gen.batch_index<=102:
            print('skipping batch:',gen.batch_index)
            continue #first batch already saved
        
        model.load_weights(filepath=weights_path+'/model.hdf5')  
        if args.full_standard:
            pred_probs, fmaps = model(x_batch_test,y_batch_test)#with eager
        else:
            pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(x_batch_test,y_batch_test)#with eager
       
        original_prob = np.max(pred_probs,axis=1)
        class_specific_prob = pred_probs#[0][np.argmax(y_batch_test,axis=1)].numpy()
        

        class_specific_matrix,class_specific_stats,class_specific_delta_matrix = save_filter_importance_batch(model,weights_path,x_batch_test,y_batch_test,label_map,original_prob,class_specific_prob,args)
        
        for i in range(len(x_batch_test)):
            #class-specific
            img_ind = i #3 (in batch)
            actual_img_ind = i + (gen.batch_index-1)*gen.batch_size
            
            y_gt = y_batch_test[img_ind]
            
            save_path = filter_data_path+'/'+ str(np.argmax(y_gt))
            if trainSplit:
                save_path+=' - train_split'
            else:
                save_path+=' - test_split'
                
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            img_count+=1
            if os.path.exists(path=save_path+'/'+str(actual_img_ind)+'_matrix_class_specific.npy'):
                continue
            print('\n\nimg_ind:',actual_img_ind, 'img_count:',img_count)
        
            np.save(file=save_path+'/'+str(actual_img_ind)+'_matrix_class_specific.npy',arr=class_specific_matrix[i,:],allow_pickle=True)
            np.save(file=save_path+'/'+str(actual_img_ind)+'_stats_class_specific.npy',arr=class_specific_stats[i,:],allow_pickle=True)
            np.save(file=save_path+'/'+str(actual_img_ind)+'_delta_matrix_class_specific.npy',arr=class_specific_delta_matrix[i,:],allow_pickle=True)
               
       
      
#%%
if (args.save_filter_importance and False):
    class_for_analysis = 0
    print ('class for analysis: ', label_map[class_for_analysis])
    trainSplit = False# train or test split
    
    save_path = filter_data_path+'/'+ str(class_for_analysis)

    if trainSplit:
        save_path+=' - train_split'
        batches=math.ceil(train_gen.n/train_gen.batch_size)
    
        train_gen.reset() #resets batch index to 0
        gen=train_gen
    else:
        save_path+=' - test_split'
        batches=math.ceil(test_gen.n/test_gen.batch_size)
    
        test_gen.reset() #resets batch index to 0
        gen=test_gen
 
    batches=math.ceil(test_gen.n/test_gen.batch_size)
    
    test_gen.reset() #resets batch index to 0
    img_count=0
    for k in range(batches):
        
        x_batch_test,y_batch_test = next(test_gen)
        
        for i in range (len(x_batch_test)):
            img_ind = i #3 (in batch)
            actual_img_ind = i + (test_gen.batch_index-1)*test_gen.batch_size

            y_gt = y_batch_test[img_ind]
            
            if np.argmax(y_gt) == class_for_analysis:
                img_count+=1
                if os.path.exists(path=save_path+'/'+str(actual_img_ind)+'_matrix_class_specific.npy'):
                    continue
                print('\n\nimg_ind:',actual_img_ind, 'img_count:',img_count)
            else:
                continue
    
            model.load_weights(filepath=weights_path+'/model.hdf5')  
            if args.full_standard:
                pred_probs = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
            else:
                pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
            
            # plt.figure(figsize = (20,2))
            # plt.imshow(x_batch_test[img_ind].squeeze(),cmap='gray')
            # plt.axis('off')
            # plt.show()
            print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
            
            original_prob = np.max(pred_probs)
            class_specific_prob = pred_probs[0][np.argmax(y_gt)].numpy()
        
            print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',class_specific_prob*100,'%\n\n')
        
            ##%% in loop individually
            argMax_matrix,argMax_pred_class,argMax_stats,argMax_delta_matrix, class_specific_matrix,class_specific_stats,class_specific_delta_matrix = save_filter_importance(model,weights_path,x_batch_test[img_ind],y_batch_test[img_ind],label_map,original_prob,class_specific_prob,actual_img_ind,args,save_path)
            
            #plot_filter_importance(class_specific_delta_matrix,class_specific_stats)
#%%
if args.save_filter_fmap:
    class_for_analysis = 0
    print ('class for analysis: ', label_map[class_for_analysis])
    save_path = './create_training_data/fmaps/'+ str(class_for_analysis)
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    batches=math.ceil(test_gen.n/test_gen.batch_size)
    
    model.load_weights(filepath=weights_path+'/model.hdf5')    

    test_gen.reset() #resets batch index to 0
    img_count=0
    for k in range(batches):
        
        x_batch_test,y_batch_test = next(test_gen)
        
        for i in range (len(x_batch_test)):
            img_ind = i #3 (in batch)
            actual_img_ind = i + (test_gen.batch_index-1)*test_gen.batch_size

            y_gt = y_batch_test[img_ind]
            
            if np.argmax(y_gt) == class_for_analysis:
                img_count+=1
                if os.path.exists(path=save_path+'/'+str(actual_img_ind)+'_fmaps_class_specific.npy'):
                    continue
                print('\n\nimg_ind:',actual_img_ind, 'img_count:',img_count)
            else:
                continue
    
            pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
            
            #plt.figure(figsize = (20,2))
            #plt.imshow(x_batch_test[img_ind].squeeze(),cmap='gray')
            #plt.axis('off')
            #plt.show()
            print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
            
            original_prob = np.max(pred_probs)
            class_specific_prob = pred_probs[0][np.argmax(y_gt)].numpy()
        
            print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',class_specific_prob*100,'%\n\n')

            np.save(file=save_path+'/'+str(actual_img_ind)+'_fmap_class_specific.npy',arr=fmaps_x2,allow_pickle=True)
            
            plot_fmap=False
            if plot_fmap:
                fig, axs = plt.subplots(8,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
                #fig.subplots_adjust(hspace = .5, wspace=.001)
                
                axs = axs.ravel()
                
                for i in range(32):
                
                    axs[i].imshow(fmaps_x2[0,:,:,i],cmap='gray')#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
                    axs[i].axis('off')
                    
                plt.show()
          
#%% analyze_filter_importance
if args.analyze_filter_importance:
    class_for_analysis = 0
    print ('class for analysis: ', label_map[class_for_analysis])

    #save_path = './create_training_data/'+ str(class_for_analysis)
    save_path = './create_training_data/'+args.model+args.dataset+'/standard/'+ str(class_for_analysis) +' - test_split'
    
    save_path_fmap = './create_training_data/fmaps/'+ str(class_for_analysis)+' - test_split'
    batches=math.ceil(test_gen.n/test_gen.batch_size)
    
    test_gen.reset() #resets batch index to 0
    for k in range(batches):
        x_batch_test,y_batch_test = next(test_gen)

        #if test_gen.batch_index<=78:
        #    print('skipping batch:',test_gen.batch_index)
        #    continue #first batch already saved-
        for i in range (len(x_batch_test)):
            img_ind = i#3
            actual_img_ind = i + (test_gen.batch_index-1)*test_gen.batch_size

            y_gt = y_batch_test[img_ind]
            

            #load saved arrays        
            if os.path.exists(path=save_path+'/'+str(actual_img_ind)+'_delta_matrix_class_specific.npy'):
                print('img_ind:',actual_img_ind)
            else:
                continue
            
            # plt.figure(figsize = (20,2))
            # plt.imshow(x_batch_test[img_ind].squeeze(),cmap='gray')
            # plt.axis('off')
            # plt.show()
                     
            model.load_weights(filepath=weights_path+'/model.hdf5')    
            pred_probs, fmaps = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
#            pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
            print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
            print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%\n\n')
            
            #class-specific
            class_specific_matrix = np.load(file=save_path+'/'+str(actual_img_ind)+'_matrix_class_specific.npy',allow_pickle=True)
            class_specific_stats = np.load(file=save_path+'/'+str(actual_img_ind)+'_stats_class_specific.npy',allow_pickle=True)
            class_specific_delta_matrix = np.load(file=save_path+'/'+str(actual_img_ind)+'_delta_matrix_class_specific.npy',allow_pickle=True)
    
            # class_specific_fmap = np.load(file=save_path_fmap+'/'+str(actual_img_ind)+'_fmap_class_specific.npy',allow_pickle=True)
            # act_sums = np.zeros((128)) 
            # for m in range(class_specific_fmap.shape[3]):
            #     act_sums[m]=np.sum(class_specific_fmap[:,:,:,m])
                
            #print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
            
            #original_prob = np.max(pred_probs)
            #class_specific_prob = pred_probs[0][np.argmax(y_gt)].numpy()
        
            #print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',class_specific_prob*100,'%')
                
            plot_filter_importance(class_specific_delta_matrix,class_specific_stats)
            #plt.plot(act_sums), plt.title('act_sums - layer 0 (top)'),plt.show()
            if (img_ind==21 and False): #zero variance
                pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
                fig, axs = plt.subplots(32,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
                #fig.subplots_adjust(hspace = .5, wspace=.001)
                
                axs = axs.ravel()
                
                for i in range(128):
                
                    axs[i].imshow(fmaps_x2[0,:,:,i],cmap='gray')#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
                    axs[i].axis('off')
                    
                plt.show()
                test_filter_importance_in_code_method(model,weights_path,x_batch_test[img_ind],y_gt,label_map,img_ind)
#%% disable best filters in each layer
test_gen.reset()
test_gen.batch_index=0 
x_batch_test,y_batch_test = next(test_gen)

actual_img_ind = 3#3, 10,21, 27, (44 == 0 mean/variance case)

img_ind = actual_img_ind -((test_gen.batch_index-1)*32)

y_gt = y_batch_test[img_ind]

model.load_weights(filepath=weights_path+'/model.hdf5')    
pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager

plt.figure(figsize = (20,2))
plt.imshow(x_batch_test[img_ind].squeeze(),cmap='gray')
plt.axis('off')
plt.show()
         
print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
print('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%\n\n')

class_for_analysis=0
save_path = './create_training_data/'+ str(class_for_analysis)
class_specific_matrix = np.load(file=save_path+'/'+str(actual_img_ind)+'_matrix_class_specific.npy',allow_pickle=True)
class_specific_stats = np.load(file=save_path+'/'+str(actual_img_ind)+'_stats_class_specific.npy',allow_pickle=True)
class_specific_delta_matrix = np.load(file=save_path+'/'+str(actual_img_ind)+'_delta_matrix_class_specific.npy',allow_pickle=True)


plot_filter_importance(class_specific_delta_matrix,class_specific_stats)

best_indexes=np.zeros((4,2))
best_indexes[0][0] = class_specific_stats[0][3]
best_indexes[1][0] = class_specific_stats[1][3]
best_indexes[2][0] = class_specific_stats[2][3]
best_indexes[3][0] = class_specific_stats[3][3]

best_indexes[0][1] = class_specific_stats[0][2]
best_indexes[1][1] = class_specific_stats[1][2]
best_indexes[2][1] = class_specific_stats[2][2]
best_indexes[3][1] = class_specific_stats[3][2]

print(best_indexes)

original_prob = np.max(pred_probs)
class_specific_prob = pred_probs[0][np.argmax(y_gt)].numpy()

check_top_filter_importance(model,weights_path,x_batch_test[img_ind],y_batch_test[img_ind],label_map,original_prob,class_specific_prob,actual_img_ind,best_indexes,args)


#%% disable best filters from histogram in loop
#TO be completed
test_gen.reset()
test_gen.batch_index=0 
x_batch_test,y_batch_test = next(test_gen)

actual_img_ind = 3#3, 10,21, 27, (44 == 0 mean/variance case)

img_ind = actual_img_ind -((test_gen.batch_index-1)*32)

y_gt = y_batch_test[img_ind]

model.load_weights(filepath=weights_path+'/model.hdf5')    
pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager

plt.figure(figsize = (20,2))
plt.imshow(x_batch_test[img_ind].squeeze(),cmap='gray')
plt.axis('off')
plt.show()
         
print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
print('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%\n\n')

class_for_analysis=0
save_path = './create_training_data/'+ str(class_for_analysis)
class_specific_matrix = np.load(file=save_path+'/'+str(actual_img_ind)+'_matrix_class_specific.npy',allow_pickle=True)
class_specific_stats = np.load(file=save_path+'/'+str(actual_img_ind)+'_stats_class_specific.npy',allow_pickle=True)
class_specific_delta_matrix = np.load(file=save_path+'/'+str(actual_img_ind)+'_delta_matrix_class_specific.npy',allow_pickle=True)


plot_filter_importance(class_specific_delta_matrix,class_specific_stats)

best_indexes=np.zeros((4,2))
best_indexes[0][0] = class_specific_stats[0][3]
best_indexes[1][0] = class_specific_stats[1][3]
best_indexes[2][0] = class_specific_stats[2][3]
best_indexes[3][0] = class_specific_stats[3][3]

best_indexes[0][1] = class_specific_stats[0][2]
best_indexes[1][1] = class_specific_stats[1][2]
best_indexes[2][1] = class_specific_stats[2][2]
best_indexes[3][1] = class_specific_stats[3][2]

print(best_indexes)

original_prob = np.max(pred_probs)
class_specific_prob = pred_probs[0][np.argmax(y_gt)].numpy()

check_top_filter_importance(model,weights_path,x_batch_test[img_ind],y_batch_test[img_ind],label_map,original_prob,class_specific_prob,actual_img_ind,best_indexes,args)
#%% disable worst filters in each layer
test_gen.reset()
test_gen.batch_index=1 
x_batch_test,y_batch_test = next(test_gen)

actual_img_ind = 44#3, 10,21, 27, 44

img_ind = actual_img_ind -((test_gen.batch_index-1)*32)

y_gt = y_batch_test[img_ind]

model.load_weights(filepath=weights_path+'/model.hdf5')    
pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager

plt.figure(figsize = (20,2))
plt.imshow(x_batch_test[img_ind].squeeze(),cmap='gray')
plt.axis('off')
plt.show()
         
print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%\n\n')

class_for_analysis=0
save_path = './create_training_data/'+ str(class_for_analysis)
class_specific_matrix = np.load(file=save_path+'/'+str(actual_img_ind)+'_matrix_class_specific.npy',allow_pickle=True)
class_specific_stats = np.load(file=save_path+'/'+str(actual_img_ind)+'_stats_class_specific.npy',allow_pickle=True)
class_specific_delta_matrix = np.load(file=save_path+'/'+str(actual_img_ind)+'_delta_matrix_class_specific.npy',allow_pickle=True)


plot_filter_importance(class_specific_delta_matrix,class_specific_stats)

worst_indexes=np.zeros((4,2))
worst_indexes[0][0] = class_specific_stats[0][1]
worst_indexes[1][0] = class_specific_stats[1][1]
worst_indexes[2][0] = class_specific_stats[2][1]
worst_indexes[3][0] = class_specific_stats[3][1]

worst_indexes[0][1] = class_specific_stats[0][0]
worst_indexes[1][1] = class_specific_stats[1][0]
worst_indexes[2][1] = class_specific_stats[2][0]
worst_indexes[3][1] = class_specific_stats[3][0]

print(worst_indexes)

original_prob = np.max(pred_probs)
class_specific_prob = pred_probs[0][np.argmax(y_gt)].numpy()

check_top_filter_importance(model,weights_path,x_batch_test[img_ind],y_batch_test[img_ind],label_map,original_prob,class_specific_prob,actual_img_ind,worst_indexes,args)
#%% visual fmaps and gradCAMs
if args.visualize_fmaps:
    img_ind = np.random.randint(0,32)
    img_ind=14#21 #21 27
    #automobiles = 6,9
    #trucks = 11,14,23,28
    
    #mnist 16, 10, 17
    #cifar 9
    #img_ind  =9,
    #pred_probs,fmaps_x1,fmaps_x2,target_1,target_2 = model.predict(np.expand_dims(x_batch_test[img_ind],0))#without eager
    #pred_probs,fmaps_x1,fmaps_x2,target_1,target_2 = model(np.expand_dims(x_batch_test[img_ind],0))#with eager
    pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(x_batch_test,y_batch_test)#with eager
    
    
    y_gt = y_batch_test[img_ind]
    
    plt.figure(figsize = (20,2))
    plt.imshow(x_batch_test[img_ind].squeeze(),cmap='gray')
    plt.axis('off')
    plt.show()
    print('predicted: ',label_map[np.argmax(pred_probs[img_ind])], ' with prob: ',np.max(pred_probs[img_ind])*100,'%')
    print ('actual: ', label_map[np.argmax(y_gt)])
    
    # plt.imshow(fmaps_x2[img_ind,:,:,0],cmap='gray')
    # plt.axis('off')
    # plt.show()
    
    output = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,np.argmax(y_batch_test[img_ind]))
    
    plt.imshow(output)
    plt.axis('off')
    plt.show()
    #%%
    output = explainer.explain((x_batch_test,None),model,np.argmax(y_batch_test))
    
    plt.imshow(output)
    plt.axis('off')
    plt.show()
    
    #%%
    fig, axs = plt.subplots(8,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
    #fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    for i in range(32):
    
        axs[i].imshow(raw_map[img_ind,:,:,i],cmap='gray')#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
        axs[i].axis('off')
    
    #plt.title('raw_map')    
    plt.show()
    
    fig, axs = plt.subplots(8,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
    #fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    for i in range(32):
    
        axs[i].imshow(forward_1[img_ind,:,:,i],cmap='gray')#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
        axs[i].axis('off')
    
    #plt.title('templates_1')    
    plt.show()
    
    fig, axs = plt.subplots(8,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
    #fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    for i in range(32):
    
        axs[i].imshow(fmaps_x1[img_ind,:,:,i],cmap='gray')#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
        axs[i].axis('off')
    
    #plt.title('masked_fmaps_x1')    
    plt.show()
    #%%
    fig, axs = plt.subplots(8,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
    #fig.subplots_adjust(hspace = .5, wspace=.001)
    
    axs = axs.ravel()
    
    for i in range(32):
    
        axs[i].imshow(fmaps_x2[img_ind,:,:,i],cmap='gray')#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
        axs[i].axis('off')
        
    plt.show()
    
    #%% analysis
    if (args.interpretable and not args.fixed_classes):
        activations_sum = model.activation_sums_2
        class_sum =model.class_sums_2
        filter_means = activations_sum/class_sum
        check=model.filter_means
        
        filter_class = tf.argmax(filter_means,1)
        image_class  = tf.argmax(y_batch_test,1)#y_batch_train#y_batch_test
        print(filter_class)
        print(image_class)
        print(image_class[img_ind])
    
    
    
    #%% for fixed classes
    
    #
    fig, axs = plt.subplots(32,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
    #fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    for i in range(128):
    
        #axs[i].imshow(fmaps_x1[0,:,:,i],cmap='gray',vmin=0, vmax=1)#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
        axs[i].imshow(fmaps_x2[img_ind,:,:,i],cmap='gray')
        axs[i].axis('off')
    
    #plt.title('templates_1')    
    plt.show()
    fig, axs = plt.subplots(32,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
    #fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    for i in range(128):
    
        #axs[i].imshow(fmaps_x1[0,:,:,i],cmap='gray',vmin=0, vmax=1)#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
        axs[i].imshow(target_2[img_ind,:,:,i],cmap='gray')
        axs[i].axis('off')
    
    #plt.title('templates_1')    
    plt.show()