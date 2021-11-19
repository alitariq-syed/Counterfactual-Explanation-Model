# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:50:49 2020

@author: Ali
"""

import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm 
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras import backend as K

#%% manual training loop
"""with or without logits is making a difference in accuracy and loss when compared to model.fit
#why? because i have used softmax as final output layer so the network outputs probabilities instead of logits... therefore from_logits must if False in this case.
#   if i want to apply softmax after getting the output from the network, then i can use from_logits=True to compute the loss """


loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
loss_fn_pre_softmax = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


optimizer = tf.keras.optimizers.RMSprop(0.001)

train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
test_loss_metric = tf.keras.metrics.Mean(name='test_loss')

train_loss_metric_2 = tf.keras.metrics.Mean(name='l1_train_loss')
test_loss_metric_2 = tf.keras.metrics.Mean(name='l1_test_loss')

train_loss_metric_3 = tf.keras.metrics.Mean(name='pre_softmax_train_loss')
test_loss_metric_3 = tf.keras.metrics.Mean(name='pre_softmax_test_loss')

#%%
def my_l1_loss(x):
    l1 = K.cast_to_floatx(0.001)
    return l1 * tf.reduce_sum(tf.abs(x))

def my_l1_loss_pre_Softmax(x):
    l1 = K.cast_to_floatx(0.001)        
    return -l1 * tf.reduce_sum(x)
    
#%%
#@tf.function 
def train_step(x_batch_test, alter_class, combined, W):
    with tf.GradientTape(persistent=False) as tape: #persistent=False  Boolean controlling whether a persistent gradient tape is created. False by default, which means at most one call can be made to the gradient() method on this object. 
    
        alter_prediction,fmatrix,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax = combined(x_batch_test, training=True)
        
        counterfactual_loss = loss_fn(alter_class, alter_prediction)
        l1_loss = my_l1_loss(fmatrix)
        
        #pre_softmax shape = batch x num_classes
        #try BCE loss without logits, i.e. maximis this value
        
        #pre_softmax_loss = loss_fn_pre_softmax(alter_class, pre_softmax)
        pre_softmax_loss = my_l1_loss_pre_Softmax(pre_softmax[:,tf.argmax(alter_class,axis=1)[0]])
        
        combined_loss = W*counterfactual_loss + 1*l1_loss + pre_softmax_loss

    
    
    gradients = tape.gradient(combined_loss, combined.trainable_variables)
    optimizer.apply_gradients(zip(gradients, combined.trainable_variables))
      
    train_loss_metric(counterfactual_loss)
    train_loss_metric_2(l1_loss)
    train_loss_metric_3(pre_softmax_loss)
    
    train_acc_metric(alter_class, alter_prediction)
    #return x1,x2,target1,target2

@tf.function 
def test_step(x_batch_test, alter_class, combined):
    
    alter_prediction,fmatrix,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax = combined(x_batch_test, training=False)
    counterfactual_loss = loss_fn(alter_class, alter_prediction)
    l1_loss = my_l1_loss(fmatrix)
    
      
    test_loss_metric(counterfactual_loss)
    test_loss_metric_2(l1_loss)

    test_acc_metric(alter_class, alter_prediction)
    #return x1,x2,target1,target2


#%%
def train_counterfactual_net(model,weights_path, generator, train_gen,test,resume):
    #assumption: its a standrd model, not interpretable model
   
    
    model.trainable = False

    # The combined model  (stacked generator and discriminator) takes
    # noise as input => generates images => determines validity 
    #img = tf.keras.Input(shape=model.input[0].shape)
    
    img = tf.keras.Input(shape=(224,224,3))
    fmatrix = generator(img)
    
    alter_prediction,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax = model([img,fmatrix])
    
    combined = tf.keras.Model(inputs=img, outputs=[alter_prediction,fmatrix,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax])
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    combined.summary()
    
    #VOC-dataset
    #label_map = ['bird',  'cat', 'cow', 'dog', 'horse', 'sheep']
    #label_map = ['cat', 'dog'] #catvsdog dataset

    for_alter_class=True#if false--> same class, oppposite class
    for_fixed_alter_class=True
    
    #manually set according VOC dataset classes
    for_class = 3
    if for_alter_class:
        if for_fixed_alter_class:
            print('training for fixed cat class')
        else:
            print('training for alter class')
    else:
        print('training for same class')

    
    if resume:
        if for_alter_class:
            combined.load_weights(filepath=weights_path+'/counterfactual_combined_model_fixed_dog_alter_class_epoch_4_binaryCFM.hdf5')
        else:
            combined.load_weights(filepath=weights_path+'/counterfactual_combined_model_same_class_epoch_35.hdf5')
    
    batches=math.ceil(train_gen.n/train_gen.batch_size)
    train_gen.reset()
    train_gen.batch_index = 0
    
    epochs = 300
    #print('Running training for %d epocs',epochs)
    
    #default_fmatrix = tf.ones((train_gen.batch_size,generator.output.shape[1]))
    if not test:
        Weight=20 # not used
        for epoch in range(0, epochs):
          #print('Start of epoch %d' % (epoch,))
              
            if epoch<3:
                Weight =1# Weight/2 #10-epoch
                #print('weight for BCE loss:', Weight)
            else:
                Weight = 1
            
            #for step,(x_batch_train, y_batch_train) in enumerate(dataset):
            with tqdm(total=batches) as progBar:
                for step in range(batches):
                  x_batch_test, y_batch_test = next(train_gen)
                 
                  #map 2-class Gt to 6 class GT
                  y_batch_test_2=np.zeros((len(y_batch_test), 6))
                  y_batch_test_2[:,1] = y_batch_test[:,0]
                  y_batch_test_2[:,3] = y_batch_test[:,1]
                  y_batch_test=y_batch_test_2
    
                  default_fmatrix = tf.ones((len(x_batch_test),generator.output.shape[1]))
                  predictions,fmaps,_ ,_,pre_softmax= model([x_batch_test,default_fmatrix], training=False)
                  
                  #it can be wrong predictions
                  #better to choose real class prediction
                  if for_alter_class:
                      if for_fixed_alter_class:
                          alter_class = np.zeros_like(y_batch_test)
                          alter_class[:,for_class] = 1
                          #alter_class = y_batch_test#sanity check
                      else:
                          alter_class = 1-y_batch_test
                  else:
                      alter_class = y_batch_test
                  
    
                  train_step(x_batch_test, alter_class,combined,Weight)
                 
                  
                  progBar.set_description('epoch %d' % (epoch))
                  progBar.set_postfix(loss=[train_loss_metric.result().numpy(),train_loss_metric_2.result().numpy(),train_loss_metric_3.result().numpy()], acc=train_acc_metric.result().numpy())
    
                  progBar.update()
             
                #end for
            #end with
            #save model at end of each epoch combined or generator?
            if for_alter_class:
                if for_fixed_alter_class:
                    combined.save_weights(filepath=weights_path+'/counterfactual_combined_model_fixed_dog_alter_class_epoch_'+str(epoch)+'.hdf5')
                else:
                    combined.save_weights(filepath=weights_path+'/counterfactual_combined_model_alter_class_epoch_'+str(epoch)+'.hdf5')
            else:
                combined.save_weights(filepath=weights_path+'/counterfactual_combined_model_same_class_epoch_'+str(epoch)+'.hdf5')
            #generator.save_weights(filepath=weights_path+'/counterfactual_generator_model_epoch_'+str(epoch)+'.hdf5')
    
        #end epoch
    else:
        #print('Testing...')
        if for_alter_class:
            combined.load_weights(filepath=weights_path+'/counterfactual_combined_model_alter_class_epoch_29.hdf5')
        else:
            combined.load_weights(filepath=weights_path+'/counterfactual_combined_model_same_class_epoch_29.hdf5')

        batches=math.ceil(train_gen.n/train_gen.batch_size)
        
        #for step,(x_batch_train, y_batch_train) in enumerate(dataset):
        with tqdm(total=batches) as progBar:
            for step in range(batches):
              x_batch_test, y_batch_test = next(train_gen)
              
              default_fmatrix = tf.ones((len(x_batch_test),generator.output.shape[1]))
              #commenting following line results in OOM error for unknown reasons
              predictions,fmaps,_ ,_,pre_softmax= model([x_batch_test,default_fmatrix], training=False)
              if for_alter_class:
                  alter_class = 1-y_batch_test
              else:
                  alter_class = y_batch_test
              probs = test_step(x_batch_test, alter_class,combined)
              
              progBar.set_postfix(loss=[test_loss_metric.result().numpy(),test_loss_metric_2.result().numpy()], acc=test_acc_metric.result().numpy())
              progBar.update()
                
          
             # Display metrics at the end of each epoch.
            test_acc = test_acc_metric.result()
            test_loss = test_loss_metric.result()
        print('\nTest loss:', test_loss.numpy())
        print('Test accuracy:', test_acc.numpy())   
    return combined, generator
    


    
    