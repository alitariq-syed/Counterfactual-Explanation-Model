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
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Softmax, GlobalAveragePooling2D
import numpy as np
import os, sys

# from models10 import MySubClassModel
from codes.train_counterfactual_net import train_counterfactual_net
#%%

from config import args, weights_path, KAGGLE, pretrained_weights_path,resume_path
from load_data import top_activation, num_classes, train_gen, label_map
from load_base_model import base_model

#%%
np.random.seed(seed=100)

if KAGGLE:
    tf.config.run_functions_eagerly(True)
    pretrained_weights_path = "/kaggle/input/cub-train-test-official-split"

import argparse
parser = argparse.ArgumentParser(description='Interpretable CNN')
parser.add_argument('--alter_class', default = 9, type = np.int32)
args2 = parser.parse_args()

if (args.train_singular_counterfactual_net and args.choose_subclass):
    raise SystemExit("train_singular_counterfactual_net and args.choose_subclass cannot be TRUE")
    
assert(args.find_global_filters==False)
#make sure training generators are setup properly

log_path  = './logs/'+args.model+args.dataset+'/standard'


if not os.path.exists(weights_path):
    os.makedirs(weights_path)
logging = args.save_logFile # save file not required if code executed in jupyter notebook

if args.train_all_classes and KAGGLE:
    assert(args.choose_subclass==False)
    print("train for ALL classes")
    classes = num_classes
    start_class=0
else:
    classes = 1
    start_class= 0#args2.alter_class
    args.alter_class = args2.alter_class


for loop in range(start_class, classes):
    tf.keras.backend.clear_session()
    # tf.compat.v1.reset_default_graph()

    
    if args.train_all_classes and KAGGLE:
        args.alter_class = loop
        
    if logging: 
        if args.counterfactual_PP:
            sys.stdout = open(file=weights_path+"/console_output_"+str(args.alter_class)+".txt", mode="w")    
        else:
            sys.stdout = open(file=weights_path+"/console_output_PN_"+str(args.alter_class)+".txt", mode="w")    
    else: print("not saving log file")
        

    print('\n',args, '\n')


    #%% create base model
    top_filters = base_model.output_shape[3] # flters in top conv layer (512 for VGG)
    fmatrix = tf.keras.layers.Input(shape=(top_filters),name='fmatrix')
    #flag = tf.keras.layers.Input(shape=(1))

    if args.model == 'VGG16/' or args.model == 'customCNN/':
        x =  MaxPool2D(name='maxpool2')(base_model.output)
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


    model.summary()

    #load saved weights
    if args.model =='customCNN/':
        model.load_weights(filepath=pretrained_weights_path+'/mnist_classifier_weights_GAP_epoch30.hdf5')
    else:
        if args.dataset == 'mnist':
            model.load_weights(filepath=pretrained_weights_path+'/mnist_classifier_weights_epoch10.hdf5')
        else:
            model.load_weights(filepath=pretrained_weights_path+'/model_fine_tune_epoch_150.hdf5')

    print("weights loaded")

    #%% train counterfactual generation network
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

    #%%

    #sigmoid = tf.convert_to_tensor(np.random.rand(512))
    #filter_map = masking_layer(sigmoid)
    num_filters = model.output[1].shape[3]
    model.trainable = False

    if args.model == 'VGG16/' or args.model == 'customCNN/':
        x =  MaxPool2D(name='maxpool2')(base_model.output)
    elif args.model == 'resnet50/':
        x =  base_model.output
    elif args.model == 'efficientnet/':
        x =  base_model.output
    mean_fmap = GlobalAveragePooling2D()(x)

    if args.dropout:
        mean_fmap_dropout = tf.keras.layers.Dropout(0.5,seed = 111)(mean_fmap)
    else:
        mean_fmap_dropout = mean_fmap
        
    if args.counterfactual_PP:
        x = Dense(num_filters,activation='sigmoid')(mean_fmap_dropout)#kernel_regularizer='l1' #,activity_regularizer='l1'
    else:
        x = Dense(num_filters,activation='relu')(mean_fmap_dropout)

    #x = tf.keras.layers.Lambda(masking_layer)(x)
    #x = CustomLayer()(x)
    #skipping gradients
    #https://stackoverflow.com/questions/39048984/tensorflow-how-to-write-op-with-gradient-in-python

    #custom layer with custom gradients
    #https://stackoverflow.com/questions/56657993/how-to-create-a-keras-layer-with-a-custom-gradient-in-tf2-0/56658149

    thresh=0.5
    PP_filter_matrix = tf.keras.layers.ThresholdedReLU(theta=thresh)(x)

    # if not args.counterfactual_PP: #for PNs

    #     PN_layer = PN_add_layer(PP_filter_matrix.shape[1],input_dim=PP_filter_matrix.shape[1])(PP_filter_matrix)
    #     PN_layer = tf.keras.layers.ReLU()(PN_layer)
        
        
        #PN_layer = Dense(num_filters,activation='relu')(PP_filter_matrix)


    # try:
    #     counterfactual_generator
    # except:
    if args.counterfactual_PP:
        counterfactual_generator = tf.keras.Model(inputs=base_model.input, outputs= [PP_filter_matrix],name='counterfactual_model')
    else:
        counterfactual_generator = tf.keras.Model(inputs=base_model.input, outputs= [x],name='counterfactual_model')
    counterfactual_generator.summary()

    #counterfactual_generator = tf.keras.Model(inputs=model.input[0], outputs= x,name='counterfactual_model')


    #%%
    cf_epochs = args.cfe_epochs #100 #100 for MNIST, 200 for CUB
    L1_weight = args.l1_weight #2 for MNIST, 2,4,6 for CUB (default 4?)
    for_class = args.alter_class if not args.train_singular_counterfactual_net else "ALL" #0 #0-9 for MNIST, 8,9s,10,11 for CUB (default 9) or 0,1,2,3 for subset training data case
    print("threshold: ", thresh)
    print("l1 weight: ", L1_weight)
    if not args.train_singular_counterfactual_net:
        print("training CF model for alter class: ",label_map[for_class])
    else:
        print("training singular CF model for all classes")

    combined, generator = train_counterfactual_net(model,weights_path,resume_path, counterfactual_generator, train_gen,args.test_counterfactual_net, args.resume_counterfactual_net,epochs=cf_epochs,L1_weight=L1_weight,for_class=for_class,label_map=label_map,logging=logging,args=args) 
    #sys.modules[__name__].__dict__.clear()
    #os._exit(00)

    # if logging: 
    #     sys.stdout.close() #= open(file=weights_path+"/console_output_"+str(args.alter_class)+".txt", mode="w")    
            
    #tf.keras.backend.clear_session()

    # del combined
    # del generator
    # del model


    # import os
    # os._exit(00)
    # from numba import cuda
    # cuda.select_device(0)
    # cuda.close()


    # from IPython import get_ipython
    # ipython = get_ipython()
    # ipython.magic("reset -f")

    # import gc
    # gc.collect()