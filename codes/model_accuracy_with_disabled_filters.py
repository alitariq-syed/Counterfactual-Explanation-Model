# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:17:29 2021

@author: Ali
"""
import math
import sys
import numpy as np
import tensorflow as tf
from codes.support_functions import restore_original_image_from_array
import matplotlib.pyplot as plt
import os
from tf_explain_modified.core.grad_cam import GradCAM
from skimage.transform import resize
from tqdm import tqdm
from sklearn.metrics import classification_report
from load_data import label_map

test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
test_loss_metric = tf.keras.metrics.Mean(name='test_loss')

@tf.function
def test_step(model, x_batch_test ,default_fmatrix):
    pred_probs,fmaps,mean_fmaps,_ ,pre_softmax= model([x_batch_test,default_fmatrix], training=False)#with eager
    return pred_probs
    
def model_accuracy_filters(model,gen, enabled_filters, args):
    
    target_class = args.alter_class
    test_recall_metric = tf.keras.metrics.Recall(
        thresholds=None, top_k=None, class_id=target_class, name=None, dtype=None)

    

    batches=math.ceil(gen.n/gen.batch_size)
    gen.reset() #resets batch index to 0
    gen.batch_index=0
    
    
    #fileName = args.model[:-1] + '_mean_fmaps_all.npy'  
    pred_y = []
    gt_y=[]     
    with tqdm(total=batches) as progBar:
        #pass
        for step in range(batches):
          x_batch_test, y_batch_test = next(gen)
          if enabled_filters is None:
              default_fmatrix = tf.ones((len(x_batch_test), model.output[1].shape[3]))
          else:
              default_fmatrix = np.ones((len(x_batch_test), model.output[1].shape[3]))
              for i in range(len(x_batch_test)):
                  default_fmatrix[i,:] = enabled_filters
          
          pred_probs = test_step(model, x_batch_test, default_fmatrix)
          
          pred_y.append(np.argmax(pred_probs,1))
          gt_y.append(np.argmax(y_batch_test,1))
          
          test_acc_metric(y_batch_test, pred_probs)
          test_recall_metric(y_batch_test, pred_probs)
      
          progBar.set_postfix(loss=test_loss_metric.result().numpy(), acc=test_acc_metric.result().numpy(), recall=test_recall_metric.result().numpy())
          progBar.update()
        #progBar.refresh()    
      
         # Display metrics at the end of each epoch.
        test_acc = test_acc_metric.result()
        test_loss = test_loss_metric.result()
        test_recall = test_recall_metric.result()
    print('\nTest loss:', test_loss.numpy())
    print('Test accuracy:', test_acc.numpy()) 
    print('Test recall for class ',target_class,': ', test_recall.numpy()) 
    
    gt_y=np.concatenate(gt_y)
    pred_y=np.concatenate(pred_y)
    print(classification_report(gt_y,pred_y,digits=4))
    c_report = classification_report(gt_y,pred_y,digits=4,output_dict=True)
    return test_acc, test_loss, c_report
        


    
    