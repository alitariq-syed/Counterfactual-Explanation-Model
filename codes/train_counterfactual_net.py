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
from codes.support_functions import get_heatmap_only_batch
from tf_explain_modified.core.grad_cam import GradCAM
import matplotlib.pyplot as plt
import sys

#%% manual training loop
"""with or without logits is making a difference in accuracy and loss when compared to model.fit
#why? because i have used softmax as final output layer so the network outputs probabilities instead of logits... therefore from_logits must if False in this case.
#   if i want to apply softmax after getting the output from the network, then i can use from_logits=True to compute the loss """


loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
loss_fn_pre_softmax = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


#optimizer = tf.keras.optimizers.RMSprop(0.001)
optimizer = tf.keras.optimizers.SGD(lr=0.01/10, momentum=0.9)

train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
test_loss_metric = tf.keras.metrics.Mean(name='test_loss')

train_loss_metric_2 = tf.keras.metrics.Mean(name='l1_train_loss')
test_loss_metric_2 = tf.keras.metrics.Mean(name='l1_test_loss')

train_loss_metric_3 = tf.keras.metrics.Mean(name='pre_softmax_train_loss')
test_loss_metric_3 = tf.keras.metrics.Mean(name='pre_softmax_test_loss')

train_loss_metric_4 = tf.keras.metrics.Mean(name='perturb_train_loss')
test_loss_metric_4 = tf.keras.metrics.Mean(name='perturb_test_loss')

train_loss_metric_5 = tf.keras.metrics.Mean(name='sparse_filter_count_train')
test_loss_metric_5 = tf.keras.metrics.Mean(name='sparse_filter_count_test')

train_loss_metric_6 = tf.keras.metrics.Mean(name='PN_loss_train')
test_loss_metric_6 = tf.keras.metrics.Mean(name='PN_test')
PN_train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='PN_train_accuracy')
PN_test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='PN_test_accuracy')

#%%
def my_l1_loss(x,a):
    l1 = K.cast_to_floatx(a)
    return l1 * tf.reduce_sum(tf.abs(x))

def my_l1_loss_pre_Softmax(x):
    l1 = K.cast_to_floatx(0.001)        
    return -l1 * tf.reduce_sum(x)

def my_filter_count(x):
    #x = filter matrix
    non_zero_in_batch = tf.math.count_nonzero(x)
    #divide by batch size
    non_zero = non_zero_in_batch/x.shape[0]
    return non_zero   
#%%
@tf.function 
def train_step(x_batch_test, alter_class, combined, W,base_model,L1_weight,PP_mode):
    perturb_loss=False
    explainer = GradCAM()
    if PP_mode:
        default_fmatrix = tf.ones((x_batch_test.shape[0],base_model.output[1].shape[3]))#512=generator.output.shape[1]
    else:
        default_fmatrix = tf.zeros((x_batch_test.shape[0],base_model.output[1].shape[3]))#512=generator.output.shape[1]

    with tf.GradientTape(persistent=False) as tape: #persistent=False  Boolean controlling whether a persistent gradient tape is created. False by default, which means at most one call can be made to the gradient() method on this object. 
    
        alter_prediction,fmatrix,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax = combined(x_batch_test, training=True)
        
        if perturb_loss:
            output,only_heatmaps = explainer.explain((x_batch_test,None),base_model,3,image_nopreprocessed=None,fmatrix=fmatrix,image_weight=0.0)#np.argmin(y_batch_test[img_ind])
            masked_preprocessed = get_heatmap_only_batch(only_heatmaps,x_batch_test)        
            perturbed_probs, perturbed_fmaps, perturbed_mean_fmap, perturbed_modified_mean_fmap_activations,perturbed_pre_softmax = base_model([masked_preprocessed,default_fmatrix])#with eager
            #plt.imshow(masked_preprocessed), plt.axis('off'), plt.title('perturbed'),plt.show()

            perturb_loss = loss_fn(alter_class, perturbed_probs)            
        
        
        counterfactual_loss = loss_fn(alter_class, alter_prediction)
        pre_softmax_loss = my_l1_loss_pre_Softmax(pre_softmax[:,tf.argmax(alter_class,axis=1)[0]])
        l1_loss_PP = 0.0
        l1_loss_PN = 0.0
        if PP_mode:
            l1_loss_PP = L1_weight* my_l1_loss(fmatrix,0.001/1)                     
            combined_loss = W*counterfactual_loss + l1_loss_PP + pre_softmax_loss
        else:
            l1_loss_PN = my_l1_loss(fmatrix,0.001)            
            combined_loss = W*counterfactual_loss + 1*l1_loss_PN

    
        #combined_loss = W*counterfactual_loss + 1*l1_loss_PN#  #+ pre_softmax_loss #+ counterfactual_PN_loss#+ perturb_loss
        #2x for CUB
        

    
    gradients = tape.gradient(combined_loss, combined.trainable_variables,unconnected_gradients='zero')
    optimizer.apply_gradients(zip(gradients, combined.trainable_variables))
      
    train_loss_metric(counterfactual_loss)
    train_loss_metric_2(l1_loss_PP)
    train_loss_metric_3(pre_softmax_loss)
    train_loss_metric_4(perturb_loss)

    train_loss_metric_5(my_filter_count(fmatrix))
    
    train_acc_metric(alter_class, alter_prediction)
    

    train_loss_metric_6(l1_loss_PN)



#@tf.function 
def train_step_experimental(x_batch_test, alter_class, combined, W,base_model):
    perturb_loss=True
    explainer = GradCAM()
    default_fmatrix = tf.ones((len(x_batch_test),512))#512=enerator.output.shape[1]

    alter_prediction,fmatrix,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax = combined([x_batch_test], training=True) #2nd x_batch_test is dummy input in place of masked input
    output,only_heatmaps = explainer.explain((x_batch_test,None),base_model,3,image_nopreprocessed=None,fmatrix=fmatrix,image_weight=0.0)#np.argmin(y_batch_test[img_ind])
    masked_preprocessed = get_heatmap_only_batch(only_heatmaps,x_batch_test)        

    with tf.GradientTape(persistent=False) as tape: #persistent=False  Boolean controlling whether a persistent gradient tape is created. False by default, which means at most one call can be made to the gradient() method on this object. 
    
        
        #if perturb_loss:
        #perturbed_probs, perturbed_fmaps, perturbed_mean_fmap, perturbed_modified_mean_fmap_activations,perturbed_pre_softmax = base_model([masked_preprocessed,default_fmatrix])#with eager
        #perturbed_probs,per_fmatrix,per_fmaps,per_mean_fmap,per_modified_mean_fmap_activations,per_pre_softmax = combined(masked_preprocessed, training=True)
        perturbed_probs,fmatrix,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax = combined([masked_preprocessed], training=True) #2nd x_batch_test is dummy input in place of masked input

        #plt.imshow(masked_preprocessed), plt.axis('off'), plt.title('perturbed'),plt.show()

        perturb_loss = loss_fn(alter_class, perturbed_probs)
        #TODO: issue: no gradients provided for any variable when perturbed used like this             
        #perturb_loss = loss_fn(alter_class, perturbed_probs)
        counterfactual_loss = loss_fn(alter_class, alter_prediction)
        l1_loss = my_l1_loss(fmatrix)
        
        #pre_softmax shape = batch x num_classes
        #try BCE loss without logits, i.e. maximis this value
        
        pre_softmax_loss = my_l1_loss_pre_Softmax(pre_softmax[:,tf.argmax(alter_class,axis=1)[0]])
        
        #combined_loss = W*counterfactual_loss + 2*l1_loss + 1*pre_softmax_loss + perturb_loss
        combined_loss = perturb_loss

    
    
    gradients = tape.gradient(combined_loss, combined.trainable_variables)
    optimizer.apply_gradients(zip(gradients, combined.trainable_variables))
      
    train_loss_metric(counterfactual_loss)
    train_loss_metric_2(l1_loss)
    train_loss_metric_3(pre_softmax_loss)
    train_loss_metric_4(perturb_loss)

    
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
def train_counterfactual_net(model,weights_path, generator, train_gen,test,resume,epochs,L1_weight,for_class,label_map,logging,args):
    #assumption: its a standrd model, not interpretable model
   
    if args.counterfactual_PP:
        mode = '' 
        print("Training CF model for PPs")
    else:
        mode = 'PN_'
        print("Training CF model for PNs")

    #model.trainable = False
    #generator.trainable = True
    # The combined model  (stacked generator and discriminator) takes
    # noise as input => generates images => determines validity 
    #img = tf.keras.Input(shape=model.input[0].shape)
    
    experimental=False
    if experimental:
        img = tf.keras.Input(shape=model.input_shape[0][1])
        
        default_fmatrix = tf.Variable(
        initial_value=np.ones((2,512)), trainable=False)
        #default_fmatrix = tf.convert_to_tensor(np.ones((2,512)))
        original_prediction,_,_,_,_ = model([img,default_fmatrix])
    
        fmatrix = generator(img)
        
        alter_prediction,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax = model([img,fmatrix])
        
        #assume masked image
        #img2 = tf.keras.Input(shape=(224,224,3))
        #default_fmatrix = tf.ones(tf.shape(fmatrix))
        #perturbed_prediction,p_fmaps,p_mean_fmap,p_modified_mean_fmap_activations,p_pre_softmax = model([img2,default_fmatrix])
        
        
        # explainer = GradCAM()
        # default_fmatrix = tf.ones_like(fmatrix)    
        # output,only_heatmaps = explainer.explain((img,None),model,3,image_nopreprocessed=None,fmatrix=fmatrix,image_weight=0.0)#np.argmin(y_batch_test[img_ind])
        # masked_preprocessed = get_heatmap_only_batch(only_heatmaps,np.expand_dims(img,0))        
        # perturbed_prediction, perturbed_fmaps, perturbed_mean_fmap, perturbed_modified_mean_fmap_activations,perturbed_pre_softmax = model([masked_preprocessed,default_fmatrix])#with eager
        
        #img2 is for masked input
        combined = tf.keras.Model(inputs=[img], outputs=[alter_prediction,original_prediction,fmatrix,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax])
        combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        combined.summary()
    else:
        img = tf.keras.Input(shape=model.input_shape[0][1:4])
        
        fmatrix = generator(img)
        
        alter_prediction,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax = model([img,fmatrix])
        
        
        combined = tf.keras.Model(inputs=img, outputs=[alter_prediction,fmatrix,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax])#,PN_prediction])
        combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        combined.summary()
    
    #VOC-dataset
    #label_map = ['bird',  'cat', 'cow', 'dog', 'horse', 'sheep']
    #label_map = ['cat', 'dog'] #catvsdog dataset

    for_alter_class=True#if false--> same class, oppposite class
    for_fixed_alter_class=True
    
    #manually set according VOC dataset classes
    for_class = for_class
    if for_alter_class:
        if for_fixed_alter_class:
            print('training for fixed alter class: ',label_map[for_class])
        else:
            print('training for alter class')
    else:
        print('training for same class')

    
    if resume:
        if for_alter_class:
            combined.load_weights(filepath=weights_path+'/'+mode+'counterfactual_combined_model_fixed_'+str(label_map[for_class])+'_alter_class.hdf5')
        else:
            combined.load_weights(filepath=weights_path+'/counterfactual_combined_model_same_class_epoch_35.hdf5')
    
    batches=math.ceil(train_gen.n/train_gen.batch_size)
    train_gen.reset()
    train_gen.batch_index = 0
    
    epochs = epochs
    #print('Running training for %d epocs',epochs)
    
    #default_fmatrix = tf.ones((train_gen.batch_size,generator.output.shape[1]))
    if logging: interval = 10000
    else: interval = 0.1

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
            with tqdm(total=batches, file=sys.stdout,mininterval=interval) as progBar:
                for step in range(batches):
                  x_batch_test, y_batch_test = next(train_gen)
                 
                  #map 2-class Gt to 6 class GT
                  #for VOC
                  if args.choose_subclass:
                      y_batch_test_2=np.zeros((len(y_batch_test), 2))
                      y_batch_test_2[:,for_class] = y_batch_test[:,0]
                      #y_batch_test_2[:,3] = y_batch_test[:,1]
                      y_batch_test=y_batch_test_2
                  
                  if args.counterfactual_PP:
                      default_fmatrix = tf.ones((len(x_batch_test),generator.output.shape[1]))
                  else:
                      #default_fmatrix = tf.zeros((len(x_batch_test),generator.output[0].shape[1]))
                      default_fmatrix = tf.zeros((len(x_batch_test),generator.output.shape[1]))
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
                  
    
                  train_step(x_batch_test, alter_class,combined,Weight,model,L1_weight,args.counterfactual_PP)
                 
                  
                  progBar.set_description('epoch %d' % (epoch),refresh=False)
                  progBar.set_postfix(loss=[train_loss_metric.result().numpy(),train_loss_metric_2.result().numpy(),train_loss_metric_3.result().numpy(),train_loss_metric_4.result().numpy(), train_loss_metric_5.result().numpy(),train_loss_metric_6.result().numpy()], acc=train_acc_metric.result().numpy(),refresh=False)
    
                  progBar.update()
             
                #end for
            #end with
            #save model at end of each epoch combined or generator?
            if for_alter_class:#false for single image
                if for_fixed_alter_class:
                    combined.save_weights(filepath=weights_path+'/'+mode+'counterfactual_combined_model_fixed_'+str(label_map[for_class])+'_alter_class.hdf5')
                else:
                    combined.save_weights(filepath=weights_path+'/counterfactual_combined_model_alter_class_epoch_'+str(epoch)+'.hdf5')
            else:
                combined.save_weights(filepath=weights_path+'/counterfactual_combined_model_same_class_epoch_'+str(epoch)+'.hdf5')
            #generator.save_weights(filepath=weights_path+'/counterfactual_generator_model_epoch_'+str(epoch)+'.hdf5')
            
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
            train_loss_metric.reset_states()
            train_loss_metric_2.reset_states()
            train_loss_metric_3.reset_states()
            train_loss_metric_4.reset_states()
            train_loss_metric_5.reset_states()
          
        #combined.save_weights(filepath=weights_path+'/counterfactual_combined_model_fixed_dog_alter_class_single_Image_epoch_'+str(epoch)+'.hdf5')
        #end epoch
    else:
        #print('Testing...')
        if for_alter_class:
            combined.load_weights(filepath=weights_path+'/counterfactual_combined_model_alter_class_epoch_29.hdf5')
        else:
            combined.load_weights(filepath=weights_path+'/counterfactual_combined_model_same_class_epoch_29.hdf5')

        batches=math.ceil(train_gen.n/train_gen.batch_size)
        
        #for step,(x_batch_train, y_batch_train) in enumerate(dataset):
        with tqdm(total=batches, file=sys.stdout) as progBar:
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
    


    
    