# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 02:04:44 2022

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
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import math
from tf_explain_modified.core.grad_cam import GradCAM
import time

#%%
from codes.support_functions import restore_original_image_from_array
from codes.find_agreement_global_MC import find_agreement_global_MC

#%%
from config import args, weights_path, KAGGLE, pretrained_weights_path
from load_data import top_activation, num_classes, train_gen, label_map, actual_test_gen
from load_base_model import base_model
from load_CFE_model import model, load_cfe_model

#%%
np.random.seed(seed=100)


    
assert(args.find_global_filters==False) #to make sure data generators are setup properly
assert(args.counterfactual_PP)
assert(args.train_all_classes)

   
#%%

if args.dataset == 'CUB200' or args.dataset == 'BraTS' or args.dataset == 'NIST': 
    test_gen =train_gen if args.find_global_filters else actual_test_gen# train_gen#actual_test_gen
    #test_gen_nopreprocess = train_gen_nopreprocess if args.find_global_filters else actual_test_gen_nopreprocess #train_gen_nopreprocess[0]#actual_test_gen_nopreprocess
    # print("using traingen gen data") if args.find_global_filters else print("using testgen gen data")

gen=test_gen
batches=math.ceil(gen.n/gen.batch_size)
    
         
gen.reset() #resets batch index to 0

default_accuracy = True
# combined = load_cfe_model()


pred_label = []
pred_label_default = []
gt_label = []
loaded_cfe_model = -1
correct_decisions_made = 0
correct_inferred_made_false = 0
wrong_pred_still_wrong=0
total_misclassifications = 0
misclassification_identified_correctly=0
correct_inference_identified_as_misclassification=0
misclassification_not_identified=0
high_confidence_misclassification_skipped = 0

counter = 0
W = model.weights[-2]

check_misclassification_only = False #dont perform top-n candidates matching. Just check agreement with the original inferred class only

filters_loaded = False
if (default_accuracy==False and check_misclassification_only==False and args.find_global_filters==False):
    try:
        MC_filters = np.load(weights_path+"/all_class_MC_filters_class.npy")
        MC_probs = np.load(weights_path+"/all_class_MC_probs_class.npy")
        filters_loaded = True
        print("filters_loaded")
    except:
        filters_loaded = False
        
        
all_classes = True
if all_classes:
    print("checking all_classes")
    selected_classes = range(200)
else:
    print("checking selected_classes")    
    selected_classes = [9,25,108,125,170]
total_images_in_selected_classes=0
total_images_inferred_in_selected_classes=0
#assert(check_misclassification_only)
start = time.time()
for k in range(batches):
    sys.stdout.write("\r batch %i of %i" % (k, batches))    
    x_batch_test,y_batch_test = next(gen)
    
    default_fmatrix = tf.ones((len(x_batch_test),base_model.output.shape[3]))
    pred_probs,_,_,_ ,_= model([x_batch_test,default_fmatrix], training=False)
    
    # pred_probs,_,_, _, _,_ = combined(x_batch_test)

    if default_accuracy:
        pred_label.append(np.argmax(pred_probs,1))
        gt_label.append(np.argmax(y_batch_test,1))
        continue
    #%%
    for img_ind in range (len(x_batch_test)):
        y_gt = y_batch_test[img_ind]
        global_img_ind = k*gen.batch_size+img_ind

        #%% check if the gt is from one of selected classes
        if np.argmax(y_gt) in selected_classes:
            total_images_in_selected_classes+=1
        #%% check if inferred class is one of the selected classes:
        if np.argmax(pred_probs[img_ind]) in selected_classes:
            total_images_inferred_in_selected_classes+=1
        else:
            continue
        #%% check if it is an actual misclassification
        if np.argmax(pred_probs[img_ind]) !=np.argmax(y_gt): total_misclassifications+=1
        
        #%%
        # #skip high confidence predictions
        skip_high_confidence = True
        if skip_high_confidence:
            if pred_probs[img_ind][np.argmax(pred_probs[img_ind])]>0.9:
                # print("skipping high confidence prediction")
                if np.argmax(pred_probs[img_ind]) != np.argmax(y_gt):
                    high_confidence_misclassification_skipped+=1
                pred_label.append(np.argmax(pred_probs[img_ind]))
                pred_label_default.append(np.argmax(pred_probs[img_ind]))
                gt_label.append(np.argmax(y_gt))
                continue
        
        #%%
        inferred_class = np.argmax(pred_probs[img_ind])
        
        selected_probs = pred_probs[img_ind] # alter_probs #pred_probs
        top_3_candidate_classes= []
        if check_misclassification_only:
            top_k=1
        else:
            top_k=5
            
        ind = np.argpartition(selected_probs, -top_k)[-top_k:]
        ind = ind[np.argsort(selected_probs.numpy()[ind])]                
        
        for i in range(top_k):
            # print('top ',str(i+1)+' predicted: ',label_map[ind[k-1-i]], ' with prob: ',pred_probs[0][ind[k-1-i]].numpy()*100,'%')
            top_3_candidate_classes.append(ind[top_k-1-i])
        
        scores = []
        #top_3_candidate_classes.append(np.argmax(y_gt))
        for cand_class in top_3_candidate_classes:
            #find agreement with global MC of each class
            #%%
            if filters_loaded and True:
                modified_mean_fmap_activations = MC_filters[cand_class][global_img_ind]
            else:
                args.alter_class = cand_class
                alter_class = cand_class
                if loaded_cfe_model==cand_class:
                    pass
                else:
                    combined = load_cfe_model()
                    loaded_cfe_model=cand_class
                
                #combined.load_weights(filepath=weights_path+'/counterfactual_generator_model_fixed_'+str(label_map[cand_class])+'_alter_class_epochs_'+str(args.cfe_epochs)+'.hdf5')
    
                alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(np.expand_dims(x_batch_test[img_ind],0))
                modified_mean_fmap_activations=modified_mean_fmap_activations[0]
                
            #%% disabled PP prediction:
            # enabled_filters = 1- fmatrix[0]
            # dis_alter_probs, dis_fmaps, dis_mean_fmap, dis_modified_mean_fmap_activations,dis_alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),enabled_filters])#with eager
                 
            # print('\nDisabled PP prediction')
            # print( 'pred class: ',label_map[np.argmax(dis_alter_probs)], '  prob: ',dis_alter_probs[0][np.argmax(dis_alter_probs)].numpy()*100,'%')
            # print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',dis_alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')
            # print( 'alter class: ',label_map[cand_class], '  prob: ',dis_alter_probs[0][cand_class].numpy()*100,'%')

            #%%
            #metrics[recall,precision, F1, mag_similarity, class_weights_score]
            freq_thresh=0.15
            metrics = find_agreement_global_MC(freq_thresh=freq_thresh, pred_MC=modified_mean_fmap_activations, target_class=cand_class, args=args,class_name =label_map[cand_class],class_weights=W[:,cand_class] )
            use_metric = 0
            scores.append(metrics[use_metric])
        
        # print('scores', scores)
        pred_label_default.append(top_3_candidate_classes[np.argmax(scores[0])])
        gt_label.append(np.argmax(y_gt))
        
        #%%
        #load avg metrics for the inferred class
        use_avg_thresh = True
        if use_avg_thresh:
            save_folder = "./model_debugging_work/epoch_"+str(args.cfe_epochs)+"/"+label_map[top_3_candidate_classes[np.argmax(scores[0])]]+"/"
            mName = args.model[:-1]+'_'+args.dataset
            
            train_metrics = np.load(file= save_folder+mName+"_metrics_aggregate_freq_thresh_"+str(freq_thresh)+"_"+str(top_3_candidate_classes[np.argmax(scores[0])])+"_train_set.np.npy")
            thresh = np.average(train_metrics[:,use_metric])
        else:
            thresh = 0.3
        #%%
        
        if scores[0] < thresh:#0.3:
            counter+=1
            if inferred_class != np.argmax(y_gt):
                misclassification_identified_correctly+=1
            else:
                correct_inference_identified_as_misclassification+=1
        
            #%%
            pred_label.append(top_3_candidate_classes[np.argmax(scores)])
            if top_3_candidate_classes[np.argmax(scores)] != inferred_class:
                if top_3_candidate_classes[np.argmax(scores)] == np.argmax(y_gt):
                    correct_decisions_made+=1
                elif inferred_class == np.argmax(y_gt):
                    correct_inferred_made_false+=1
                else:
                    wrong_pred_still_wrong+=1
            #%%
        else:
            pred_label.append(top_3_candidate_classes[np.argmax(scores[0])])
            if inferred_class != np.argmax(y_gt):
                misclassification_not_identified+=1


        continue
end = time.time()
pred_label = np.concatenate(pred_label,axis=None)
pred_label_default = np.concatenate(pred_label_default,axis=None)
gt_label = np.concatenate(gt_label,axis=None)

acc = np.sum(pred_label_default==gt_label)/len(pred_label_default)
print("\n\ndefault acc",acc)
acc = np.sum(pred_label==gt_label)/len(pred_label)
print("altered acc",acc)

print("correct_decisions_made:", correct_decisions_made)
print("correctly_inferred_made_false:", correct_inferred_made_false)
print("wrong_pred_still_wrong:", wrong_pred_still_wrong)

print("\n\ntotal_images_in_selected_classes: ",total_images_in_selected_classes)
print("total_images_inferred_in_selected_classes: ",total_images_inferred_in_selected_classes)
print("total_misclassifications: ", total_misclassifications)
print("misclassification_identified_correctly: ", misclassification_identified_correctly)
print("correct_inference_identified_as_misclassification: ", correct_inference_identified_as_misclassification)
print("misclassification_not_identified: ", misclassification_not_identified)
print("high_confidence_misclassification_skipped: ", high_confidence_misclassification_skipped)
print("time taken: ",end - start)

print("\ncounter ", counter)