# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 22:15:30 2022

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
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import optimizers
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm # to monitor progress
import numpy as np
import os, sys
import math

# from models10 import MySubClassModel
from codes.train_counterfactual_net import train_counterfactual_net
#%%

from config import args, weights_path, KAGGLE, pretrained_weights_path
from load_data import top_activation, num_classes, train_gen, test_gen
from load_base_model import base_model

from codes.model_accuracy_with_disabled_filters import model_accuracy_filters

if args.dataset=='CUB200': 
    from load_data import actual_test_gen
else:
    actual_test_gen = test_gen
#%%
np.random.seed(seed=100)

# if KAGGLE:
#     tf.config.run_functions_eagerly(True)
#     pretrained_weights_path = "/kaggle/input/cub-train-test-official-split"

    
assert(args.find_global_filters==False)
#make sure training generators are setup properly

if not os.path.exists(weights_path):
    os.makedirs(weights_path)


#%% create base model
top_filters = base_model.output_shape[3] # flters in top conv layer (512 for VGG)
fmatrix = tf.keras.layers.Input(shape=(top_filters),name='fmatrix')

#set last conv layer as trainable to encourage MC filter activation/model debugging
base_model.layers[-1].trainable = True
# base_model.trainable = True

if args.model == 'VGG16/' or args.model == 'myCNN/':
    x =  MaxPool2D()(base_model.output)
elif args.model == 'resnet50/':
    x =  base_model.output
elif args.model == 'efficientnet/':
    x =  base_model.output
mean_fmap = GlobalAveragePooling2D()(x)
dropout = tf.keras.layers.Dropout(0.5,seed = 111)(mean_fmap)
#%%
# x = tf.keras.layers.Activation('sigmoid')(mean_fmap)
# fmatrix = tf.keras.layers.ThresholdedReLU(theta=0.5)(x) #approx binary to make learnable
#modified_fmap = mean_fmap*fmatrix

#%%

pre_softmax = Dense(num_classes,activation=None)(dropout)
out = tf.keras.layers.Activation(top_activation)(pre_softmax)

model = tf.keras.Model(inputs=[base_model.input], outputs= [out],name='base_model')

model.compile(optimizer=optimizers.SGD(lr=0.001/10, momentum = 0.9), 
                  loss=['categorical_crossentropy'], 
                  metrics=['accuracy'])

model.summary()

#load saved weights
if args.model =='myCNN/':
    model.load_weights(filepath=pretrained_weights_path+'/model_transfer_epoch_50.hdf5')
else:
    if args.fine_tune:
        #model.load_weights(filepath=pretrained_weights_path+'/model_debugged.hdf5')
        model.load_weights(filepath=weights_path+'/model_retrained_normal.hdf5')
    else:
        model.load_weights(filepath=pretrained_weights_path+'/model_fine_tune_epoch_150.hdf5')
    # model.load_weights(filepath=pretrained_weights_path+'/model_debugged_epoch_9.hdf5')

print("weights loaded")


save_path = weights_path+'/model_retrained_normal.hdf5'
checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max', save_weights_only = True)
callbacks_list = [checkpoint]
 
#%%
if args.train:
    history = model.fit(train_gen, epochs=20, verbose=1, callbacks=callbacks_list, validation_data=test_gen, shuffle=True)
        
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

#%%
if args.test:
    #load best weights

    #model.load_weights(filepath=weights_path+'/model_retrained_normal.hdf5')
    
    #model.evaluate(actual_test_gen,verbose=1)
         
    pred_probs= model.predict(actual_test_gen,verbose=1)
    
    pred_classes = np.argmax(pred_probs,1)
    #actual_classes = np.argmax(test_gen.classes,1)
    actual_classes = actual_test_gen.classes
    print(confusion_matrix(actual_classes,pred_classes))
    print(classification_report(actual_classes,pred_classes,digits=4)) 
