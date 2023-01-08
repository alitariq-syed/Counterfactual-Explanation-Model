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
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import optimizers
from tensorflow.python.keras import backend as K
from tqdm import tqdm # to monitor progress
import numpy as np
import os, sys
import math

# from models10 import MySubClassModel
from codes.train_counterfactual_net import train_counterfactual_net
#%%

from config import args, weights_path, KAGGLE, pretrained_weights_path
from load_data import top_activation, num_classes, train_gen, label_map, test_gen, actual_test_gen
from load_base_model import base_model

from codes.model_accuracy_with_disabled_filters import model_accuracy_filters

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
#fmatrix = tf.keras.layers.Input(shape=(top_filters),name='fmatrix')

#set last conv layer as trainable to encourage MC filter activation/model debugging
base_model.layers[-1].trainable = True
#base_model.trainable = True

if args.model == 'VGG16/' or args.model == 'myCNN/':
    x =  MaxPool2D()(base_model.output)
elif args.model == 'resnet50/':
    x =  base_model.output
elif args.model == 'efficientnet/':
    x =  base_model.output
mean_fmap = GlobalAveragePooling2D()(x)
# dropout = tf.keras.layers.Dropout(0.5,seed = 111)(mean_fmap)
dropout = tf.keras.layers.Dropout(0.5)(mean_fmap)
#%%
x = tf.keras.layers.Activation('sigmoid')(mean_fmap)
#x = Dense(512,activation='sigmoid')(mean_fmap)
fmatrix = tf.keras.layers.ThresholdedReLU(theta=0.5)(x) #approx binary to make learnable
#modified_fmap = mean_fmap*fmatrix

#%%

pre_softmax = Dense(num_classes,activation=None)(dropout)
out = tf.keras.layers.Activation(top_activation)(pre_softmax)

model = tf.keras.Model(inputs=[base_model.input], outputs= [out, mean_fmap, fmatrix,pre_softmax,],name='base_model')

#%%
# model.compile(optimizer=optimizers.SGD(lr=0.001/10, momentum = 0.9), 
#                   loss=['categorical_crossentropy'], 
#                   metrics=['accuracy'])
#%%
model.summary()

#load saved weights
if args.model =='myCNN/':
    model.load_weights(filepath=pretrained_weights_path+'/model_transfer_epoch_50.hdf5')
else:
    if args.fine_tune:
        #model.load_weights(filepath=pretrained_weights_path+'/model_debugged.hdf5')
        model.load_weights(filepath=weights_path+'/model_debugged.hdf5')
    else:
        model.load_weights(filepath=pretrained_weights_path+'/model_fine_tune_epoch_150.hdf5')
        # model.load_weights(filepath=pretrained_weights_path+'/model_debugged_0.70240325_7042.hdf5')
        
    # model.load_weights(filepath=pretrained_weights_path+'/model_debugged_epoch_9.hdf5')

print("weights loaded")

 
#%%
if args.dataset == 'cxr1000':
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    train_acc_metric = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    test_acc_metric = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
else:
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    
optimizer = optimizers.SGD(lr=0.001/10, momentum = 0.9)#optimizers.RMSprop(0.001)

train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
MC_filters_loss = tf.keras.metrics.Mean(name='MC_filters_loss')
non_MC_filters_loss = tf.keras.metrics.Mean(name='non_MC_filters_loss')
filter_count = tf.keras.metrics.Mean(name='filter_count')
pre_softmax_loss = tf.keras.metrics.Mean(name='pre_softmax_loss')
test_loss_metric = tf.keras.metrics.Mean(name='test_loss')

#%%
def my_filter_count(x):
    #x = filter matrix
    non_zero_in_batch = tf.math.count_nonzero(x,1,dtype='float32')
    non_zero = tf.math.reduce_mean(non_zero_in_batch)
    return non_zero 

def my_l1_loss_MC(x,a):
    l1 = K.cast_to_floatx(a)
    return l1 * tf.reduce_sum(tf.abs(x))
def my_l1_loss_nonMC(x,a):
    l1 = K.cast_to_floatx(a)
    return l1 * tf.reduce_sum(tf.abs(x))
def my_filter_count(x):
    #x = filter matrix
    non_zero_in_batch = tf.math.count_nonzero(x,1,dtype='float32')
    non_zero = tf.math.reduce_mean(non_zero_in_batch)
    return non_zero   
def my_l1_loss_pre_Softmax(x,a):
    l1 = K.cast_to_floatx(a)        
    return -l1 * tf.reduce_sum(x)
#%%

@tf.function 
def train_step(images, labels, global_MC_filters):
  with tf.GradientTape(persistent=False) as tape: #persistent=False  Boolean controlling whether a persistent gradient tape is created. False by default, which means at most one call can be made to the gradient() method on this object. 
    
    # training=True is osnly needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
       
    predictions, mean_fmap, fmatrix, pre_softmax = model([images], training=True)   

    crossentropy_loss = loss_fn(labels, predictions)
    
    # mean_fmap_binary = tf.where(mean_fmap>0,1.0,0.0)
    #x = tf.keras.layers.Activation('sigmoid')(mean_fmap)
    #mean_fmap_binary = tf.keras.layers.ThresholdedReLU(theta=0.5)(mean_fmap) #approx binary to make learnable
    
    
    L1_weight=1
    l1_loss_MC_filters = -L1_weight* my_l1_loss_MC(fmatrix*global_MC_filters,0.0001*10)                     
    l1_loss_non_MC_filters = L1_weight* my_l1_loss_nonMC(fmatrix*(1-global_MC_filters),0.00001*2)   
    
    # l1_loss_MC_filters = -L1_weight* my_l1_loss_pre_Softmax(tf.matmul(tf.transpose(mean_fmap*global_MC_filters),pre_softmax),0.001/1)                     
    # l1_loss_non_MC_filters = L1_weight* my_l1_loss_pre_Softmax(tf.matmul(tf.transpose(mean_fmap*(1-global_MC_filters)),pre_softmax),0.001/1)   
    
    # l1_loss_MC_filters = L1_weight* my_l1_loss_MC(mean_fmap-global_MC_filters,0.00001/1)                     
    # l1_loss_non_MC_filters = L1_weight* my_l1_loss_nonMC(mean_fmap*(1-global_MC_filters),0.00001/1)   
    
    # l1_loss_MC_filters = tf.keras.losses.MAE(global_MC_filters,fmatrix)
    # l1_loss_non_MC_filters = tf.keras.losses.MAE(1-global_MC_filters,fmatrix)
    
    #%%
    # pre_softmax_logits=[]
    # for i in range(len(pre_softmax)):
    #     logits = pre_softmax[i,tf.argmax(labels,axis=1)[i]]
    #     pre_softmax_logits.append(logits)
    # pre_softmax_logits = tf.convert_to_tensor(pre_softmax_logits)
    
    loss_pre_softmax=0.0
    # loss_pre_softmax = my_l1_loss_pre_Softmax(pre_softmax_logits,0.001)
    #%%
    # pred_MC_binary = tf.where(mean_fmap>0,1.0,0.0)
    # filter_histogram_cf_binary = global_MC_filters

    # recall = -tf.reduce_sum(pred_MC_binary * filter_histogram_cf_binary,1)/tf.reduce_sum(filter_histogram_cf_binary,1)
    # precision = tf.reduce_sum(pred_MC_binary * filter_histogram_cf_binary,1)/tf.reduce_sum(pred_MC_binary,1)
    #avg_prec_recall = (precision+precision)/2
    #F1 = -2 * (precision * recall) / (precision + recall+0.00000001)                  
    #%%
    
    combined_loss = 1*crossentropy_loss + (l1_loss_MC_filters + l1_loss_non_MC_filters)# + l1_loss_non_MC_filters)*1
  
  gradients = tape.gradient(combined_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  train_loss_metric(crossentropy_loss)
  MC_filters_loss(l1_loss_MC_filters)
  non_MC_filters_loss(l1_loss_non_MC_filters)
  pre_softmax_loss(loss_pre_softmax)
  filter_count(my_filter_count(mean_fmap))

  train_acc_metric(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  default_fmatrix = tf.ones((len(images),512))
  predictions, mean_fmap, fmatrix, pre_softmax = model([images],training = False)

  loss_value = loss_fn(labels, predictions)

  test_loss_metric(loss_value)
  test_acc_metric(labels, predictions)
  return predictions

#%% load global MC filters for each class
global_MC_filters = []
for target_class in range(num_classes):
    class_name = label_map[target_class]
    #print(class_name)
    save_folder = "./model_debugging_work/epoch_"+str(args.cfe_epochs)+"/"+class_name+"/"
    
    mName = args.model[:-1]+'_'+args.dataset
    
    try:
        filter_magnitude_cf = np.load(file= save_folder+mName+"_normalized_filter_magnitude_cf_"+str(target_class)+"_train_set.np.npy")
        filter_histogram_cf = np.load(file= save_folder+mName+"_filter_histogram_cf_"+str(target_class)+"_train_set.np.npy")
    except:
        raise Exception("Global MC Filters not loaded") 
        #return 0  
    
    # if enable_prints:
    #     plt.plot(filter_histogram_cf), plt.ylim([0, np.max(filter_histogram_cf)+1]),plt.xlabel("Filter number"),plt.ylabel("Filter activation count"), plt.show()
    #     plt.plot(filter_magnitude_cf/max(filter_histogram_cf)),plt.xlabel("Filter number"),plt.ylabel("Avg. activation magnitude"), plt.ylim([0, np.max(filter_magnitude_cf/np.max(filter_histogram_cf))+1]), plt.show()


    #     plt.plot(pred_MC), plt.ylim([0, np.max(pred_MC)+1]),plt.xlabel("Filter number"),plt.ylabel("Filter activation magnitude"), plt.show()
 
    freq_thresh = 0.0 #percent
    filter_histogram_cf_binary = tf.where(filter_histogram_cf>(freq_thresh*max(filter_histogram_cf)),1.0,0.0)
    filter_magnitude_cf_thresholded = (filter_magnitude_cf*filter_histogram_cf_binary)/max(filter_histogram_cf)
    global_MC_filters.append(filter_histogram_cf_binary)
global_MC_filters = np.vstack(global_MC_filters)

#%%
# Iterate over the batches of the dataset.
#for step, (x_batch_train, y_batch_train) in enumerate(train_gen):
train_gen.reset()
train_gen.batch_index=0
test_gen.reset()
test_gen.batch_index=0

best_val_accuracy = 0
if args.train:
    epochs = 20
    batches=math.ceil(train_gen.n/train_gen.batch_size)
    test_batches=math.ceil(test_gen.n/test_gen.batch_size)

    #print('Running training for %d epocs',epochs)
    for epoch in range(epochs):
      #print('Start of epoch %d' % (epoch,))
      
      #for step,(x_batch_train, y_batch_train) in enumerate(dataset):
      with tqdm(total=batches) as progBar:
          for step in range(batches):
            x_batch_train, y_batch_train = next(train_gen)
            
            gt_class = np.argmax(y_batch_train,1)

            train_step(x_batch_train, y_batch_train,global_MC_filters[gt_class])

            progBar.set_description('epoch %d' % (epoch))
            progBar.set_postfix(loss=[train_loss_metric.result().numpy(),MC_filters_loss.result().numpy(),non_MC_filters_loss.result().numpy(),filter_count.result().numpy(),pre_softmax_loss.result().numpy()], acc=train_acc_metric.result().numpy())
            progBar.update()
            #break
    
          #epoch end
          # model.save_weights(filepath=weights_path+'/model_debugged_epoch_'+str(epoch)+'.hdf5')
          model.save_weights(filepath=weights_path+'/model_debugged.hdf5')
  
           # Display metrics at the end of each epoch.
          train_acc = train_acc_metric.result()
          train_loss = train_loss_metric.result()
        
          # Reset training metrics at the end of each epoch
          train_acc_metric.reset_states()
          train_loss_metric.reset_states()
          MC_filters_loss.reset_states()
          non_MC_filters_loss.reset_states()
          pre_softmax_loss.reset_states()
          filter_count.reset_states()

          #break
          
      #evaluate model at end of epoch
      evaluate=True
      if evaluate and (epoch%1==0):
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
          
          if test_acc.numpy()>best_val_accuracy:
              best_val_accuracy = test_acc.numpy()
              print("best_val_accuracy (saving model):", best_val_accuracy)
              model.save_weights(filepath=weights_path+'/model_debugged_'+str(best_val_accuracy)+'.hdf5')



if args.test:
    print('Testing...')

    #model.load_weights(weights_path)
    gen = actual_test_gen #test_gen # train_gen #actual_test_gen
    batches=math.ceil(gen.n/gen.batch_size)

    #for step,(x_batch_train, y_batch_train) in enumerate(dataset):
    with tqdm(total=batches) as progBar:
        #pass
        for step in range(batches):
          x_batch_test, y_batch_test = next(gen)
          
          probs = test_step(x_batch_test, y_batch_test)
          
          progBar.set_postfix(loss=test_loss_metric.result().numpy(), acc=test_acc_metric.result().numpy(), refresh=False)
          progBar.update()
        #progBar.refresh()    
      
         # Display metrics at the end of each epoch.
        test_acc = test_acc_metric.result()
        test_loss = test_loss_metric.result()
    print('\nTest loss:', test_loss.numpy())
    print('Test accuracy:', test_acc.numpy()) 
    # model.save_weights(filepath=weights_path+'/model_debugged_'+str(test_acc.numpy())+'.hdf5')
#sys.exit()
# enabled_filters = np.ones(512)
# test_acc, test_loss, c_report = model_accuracy_filters(model,actual_test_gen, enabled_filters, args)