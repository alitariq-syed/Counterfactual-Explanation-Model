# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 23:50:34 2022

@author: Ali
"""
#%%
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Softmax, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import optimizers
import numpy as np

#%%

label_map = np.loadtxt(fname='G:/CUB_200_2011/CUB_200_2011/classes.txt',dtype='str')
label_map = label_map[:,1].tolist()
#%% create base model

vgg = VGG16(weights='imagenet',include_top = False,input_shape=(224,224,3))

base_model = tf.keras.Model(vgg.input,vgg.layers[-2].output)

#%% classifier model
x =  MaxPool2D()(base_model.output)
mean_fmap = GlobalAveragePooling2D()(x)
dropout = tf.keras.layers.Dropout(0.5)(mean_fmap)
out = Dense(200,activation='softmax')(dropout)

model = tf.keras.Model(inputs=[base_model.input], outputs= [out],name='base_model')

model.compile(optimizer=optimizers.SGD(lr=0.001/10, momentum = 0.9), 
                  loss=['categorical_crossentropy'], 
                  metrics=['accuracy'])

model.summary()

model.load_weights(filepath='model_fine_tune_epoch_150.hdf5')

#%% create base model
top_filters = base_model.output_shape[3] # flters in top conv layer (512 for VGG)
fmatrix = tf.keras.layers.Input(shape=(top_filters),name='fmatrix')
#flag = tf.keras.layers.Input(shape=(1))

x =  MaxPool2D()(base_model.output)
mean_fmap = GlobalAveragePooling2D()(x)


#modify base model (once it has been pre-trained separately) to be used with CF model later
# if args.counterfactual_PP:
modified_fmap = mean_fmap*fmatrix
# else:#PN
    # modified_fmap = mean_fmap+fmatrix
pre_softmax = Dense(200,activation=None)(modified_fmap)
out = tf.keras.layers.Activation('softmax')(pre_softmax)
model_modified = tf.keras.Model(inputs=[base_model.input, fmatrix], outputs= [out,base_model.output, mean_fmap, modified_fmap,pre_softmax],name='base_model')

# if args.counterfactual_PP:
# default_fmatrix = tf.ones((train_gen.batch_size,base_model.output.shape[3]))
# else:
    # default_fmatrix = tf.zeros((train_gen.batch_size,base_model.output.shape[3]))


#model.summary()

#load saved weights
model_modified.load_weights(filepath='/model_fine_tune_epoch_150.hdf5')
print("base model weights loaded")

#%% create CFE model
num_filters = model.output[1].shape[3]
model_modified.trainable = False

x =  MaxPool2D()(base_model.output)
mean_fmap = GlobalAveragePooling2D()(x)

# if args.counterfactual_PP:
x = Dense(num_filters,activation='sigmoid')(mean_fmap)#kernel_regularizer='l1' #,activity_regularizer='l1'
# else:
    # x = Dense(num_filters,activation='relu')(mean_fmap)


thresh=0.5
PP_filter_matrix = tf.keras.layers.ThresholdedReLU(theta=thresh)(x)



counterfactual_generator = tf.keras.Model(inputs=base_model.input, outputs= [PP_filter_matrix],name='counterfactual_model')


def load_cfe_model(selected_class):

    counterfactual_generator.load_weights(filepath='contrastive model weights/counterfactual_generator_model_fixed_'+str(label_map[selected_class])+'_alter_class_epochs_200.hdf5')
        
    model.trainable = False
    img = tf.keras.Input(shape=model.input_shape[0][1:4])

    fmatrix = counterfactual_generator(img)
    
    #binarization here is not reducing loss during training. So only use it for test time and not for training
    fmatrix = tf.where(fmatrix > 0, 1.0, 0.0)

    alter_prediction,fmaps,mean_fmap, modified_mean_fmap_activations,pre_softmax = model([img,fmatrix])
    
    combined = tf.keras.Model(inputs=img, outputs=[alter_prediction,fmatrix,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax])
    

    return combined