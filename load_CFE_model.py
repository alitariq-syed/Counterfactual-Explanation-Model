# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 23:55:58 2022

@author: Ali
"""

#%%
import tensorflow as tf
print('TF version: ', tf.__version__)

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Softmax, GlobalAveragePooling2D

#%%
from config import args, weights_path, pretrained_weights_path
from load_data import top_activation, num_classes,label_map, train_gen
from load_base_model import base_model


#%% create base model
top_filters = base_model.output_shape[3] # flters in top conv layer (512 for VGG)
fmatrix = tf.keras.layers.Input(shape=(top_filters),name='fmatrix')
#flag = tf.keras.layers.Input(shape=(1))

if args.model == 'VGG16/' or args.model == 'myCNN/':
    x =  MaxPool2D()(base_model.output)
elif args.model == 'resnet50/':
    x =  base_model.output
elif args.model == 'efficientnet/':
    x =  base_model.output
mean_fmap = GlobalAveragePooling2D()(x)


#modify base model (once it has been pre-trained separately) to be used with CF model later
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


#model.summary()

#load saved weights
if args.model =='myCNN/':
    model.load_weights(filepath=pretrained_weights_path+'/model_transfer_epoch_50.hdf5')
else:
    model.load_weights(filepath=pretrained_weights_path+'/model_fine_tune_epoch_150.hdf5')

print("base model weights loaded")

#%% create CFE model
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


thresh=0.5
PP_filter_matrix = tf.keras.layers.ThresholdedReLU(theta=thresh)(x)



counterfactual_generator = tf.keras.Model(inputs=base_model.input, outputs= [PP_filter_matrix],name='counterfactual_model')


def load_cfe_model():
    if not args.train_singular_counterfactual_net:
        if args.choose_subclass:
            counterfactual_generator.load_weights(filepath=weights_path+'/counterfactual_generator_model_only_010.Red_winged_Blackbird_alter_class_epochs_'+str(args.cfe_epochs)+'.hdf5')
        else:                
            if args.counterfactual_PP:
                mode = '' 
                # print("Loading CF model for PPs")
            else:
                mode = 'PN_'
                print("Loading CF model for PNs")
            counterfactual_generator.load_weights(filepath=weights_path+'/'+mode+'counterfactual_generator_model_fixed_'+str(label_map[args.alter_class])+'_alter_class_epochs_'+str(args.cfe_epochs)+'.hdf5')
        
    model.trainable = False
    img = tf.keras.Input(shape=model.input_shape[0][1:4])

    fmatrix = counterfactual_generator(img)
    
    #binarization here is not reducing loss during training. So only use it for test time and not for training
    fmatrix = tf.where(fmatrix > 0, 1.0, 0.0)

    alter_prediction,fmaps,mean_fmap, modified_mean_fmap_activations,pre_softmax = model([img,fmatrix])
    
    combined = tf.keras.Model(inputs=img, outputs=[alter_prediction,fmatrix,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax])
    
    if args.train_singular_counterfactual_net:
        combined.load_weights(filepath=weights_path+'/counterfactual_combined_model_ALL_classes_epoch_131.hdf5')

    return combined