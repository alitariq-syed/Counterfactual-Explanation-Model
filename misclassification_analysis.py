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
from load_data import top_activation, num_classes, train_gen, label_map,actual_test_gen
from load_base_model import base_model
from load_CFE_model import model, load_cfe_model,fmatrix

#%%
np.random.seed(seed=100)
    
assert(args.find_global_filters==False) #to make sure data generators are setup properly
assert(args.counterfactual_PP)
# assert(args.train_all_classes)# why needed? cannot remember

#%%
W = model.weights[-2]
act_threshold=-0.15
plt.plot(W.numpy()), plt.title('Weights'), plt.legend(['cat','dog']),
plt.hlines(act_threshold,xmin=0,xmax=512,colors='r'),plt.show()

important_filter_weights_1 = np.where(W[:,0]<=act_threshold)
important_filter_weights_2 = np.where(W[:,1]<=act_threshold)
    
#%%
explainer = GradCAM()
misclassification_analysis = False


combined = load_cfe_model()

class_for_analysis = args.analysis_class#args.analysis_class#9#9 170#np.random.randint(200)#23#11 #cat for VOC dataset
#alter_class=args.alter_class
print ('class for analysis: ', label_map[class_for_analysis])
#print ('alter class: ', label_map[alter_class])

weights_alter_class = W[:,args.analysis_class]
plt.plot(weights_alter_class),plt.title("weight alter class "+str(args.analysis_class)),plt.show()

# if args.dataset == 'CUB200' or args.dataset == 'BraTS' or args.dataset == 'NIST': 
test_gen =train_gen if args.find_global_filters else actual_test_gen# train_gen#actual_test_gen
    #test_gen_nopreprocess = train_gen_nopreprocess if args.find_global_filters else actual_test_gen_nopreprocess #train_gen_nopreprocess[0]#actual_test_gen_nopreprocess
    # print("using traingen gen data") if args.find_global_filters else print("using testgen gen data")

gen=test_gen
batches=math.ceil(gen.n/gen.batch_size)



filter_histogram_cf = tf.zeros(fmatrix[0].shape[0])
filter_magnitude_cf = tf.zeros(fmatrix[0].shape[0])
filter_sum = 0


gen.reset() #resets batch index to 0
local_misclassifications = 0
img_count=0
if args.dataset=='mnist':
    alter_class_images_count = np.sum(np.argmax(gen.y,1)==args.analysis_class)
    alter_class_starting_batch = np.floor(np.where(np.argmax(gen.y,1)==args.analysis_class)[0][0]/gen.batch_size).astype(np.int32)
else:
    alter_class_images_count = np.sum(gen.classes==args.analysis_class)
    alter_class_starting_batch = np.floor(np.where(gen.labels==args.analysis_class)[0][0]/gen.batch_size).astype(np.int32)
index_reached=0
loaded_cfe_model=-1
start = time.time()
for k in range(batches):

    if args.find_global_filters:
        if k < alter_class_starting_batch:
            sys.stdout.write("\rskipping batch %i of %i" % (k, batches))
            continue
        else:
            x_batch_test,y_batch_test = next(gen)                
            gen.batch_index = k

        
    x_batch_test,y_batch_test = next(gen)
    
    default_fmatrix = tf.ones((len(x_batch_test),base_model.output.shape[3]))

    if args.find_global_filters:
        if gen.batch_index < alter_class_starting_batch and gen.batch_index >0:
            continue

    
    sys.stdout.write("\rbatch %i of %i" % (k, batches))
    sys.stdout.flush()
    
    for i in range (len(x_batch_test)):
        img_ind = i#3

        if gen.batch_index==0:
            actual_img_ind = i + (batches-1)*gen.batch_size
        else:
            actual_img_ind = i + (gen.batch_index-1)*gen.batch_size
        
        if misclassification_analysis:
            if actual_img_ind != 240:#240, 241 wrong for class 9; 655 for 25
                print('skipping img_ind:',actual_img_ind)
                continue
                

        y_gt = y_batch_test[img_ind]
        
         #skip other class        
        if class_for_analysis==np.argmax(y_gt):
            pass
            # print('\n\nimg_ind:',actual_img_ind)
        else:
            continue
        
        #%%
        #model.load_weights(filepath=weights_path+'/model.hdf5')  
        pred_probs,fmaps,mean_fmaps,_ ,pre_softmax= model([np.expand_dims(x_batch_test[img_ind],0),np.expand_dims(default_fmatrix[0],0)], training=False)#with eager
        #pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
        print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
        print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%')
        #print('pre_softmax: ',pre_softmax[0][0:10])
        
       
        
        #wrong prediction
        if np.argmax(pred_probs) != np.argmax(y_gt) and True:
            print("wrong prediction")
            # incorrect_class=np.argmax(pred_probs)
            # print("skipping wrong prediction")
            # continue
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
        
        #%%
        inferred_class = np.argmax(pred_probs[0])
        args.alter_class = inferred_class
        alter_class = args.alter_class
        alter_class = inferred_class
        if loaded_cfe_model==inferred_class:
            pass
        else:
            combined = load_cfe_model()
            loaded_cfe_model=inferred_class
        
        #%%            
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
        if args.dataset != 'mnist':
            image_nopreprocessed = restore_original_image_from_array(x_batch_test[img_ind].squeeze())
        else:
            image_nopreprocessed = x_batch_test[img_ind]
        if gradcam:
            output_orig,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,np.argmax(pred_probs),image_nopreprocessed=np.expand_dims(image_nopreprocessed,0),fmatrix=default_fmatrix)
            
            plt.imshow(output_orig), plt.axis('off'), plt.title('original prediction')
            plt.show()
            
            output_gradcam_alter,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,alter_class,image_nopreprocessed=np.expand_dims(image_nopreprocessed,0),fmatrix=default_fmatrix)
            plt.imshow(output_gradcam_alter), plt.axis('off'), plt.title('GradCAM alter prediction')
            plt.show()
        original_mean_fmap_activations = GlobalAveragePooling2D()(fmaps)
        #plt.plot(original_mean_fmap_activations[0]), plt.title('original_mean_fmap_activations'),plt.show()
        plt.plot(mean_fmaps[0]), plt.ylim([0, np.max(mean_fmaps)+1]), plt.title('mean_fmaps'),plt.show()
       
        
        #combined.load_weights(filepath=weights_path+'/counterfactual_combined_model_fixed_'+str(label_map[incorrect_class])+'_alter_class.hdf5')
        
        #fmatrix = counterfactual_generator(np.expand_dims(x_batch_test[img_ind],0))
        if args.counterfactual_PP:
            alter_probs,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(np.expand_dims(x_batch_test[img_ind],0))
        else:
            alter_probs,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax,PN_add = combined(np.expand_dims(x_batch_test[img_ind],0))
        
        #alter_probs, c_fmaps, c_mean_fmap, c_modified_mean_fmap_activations,alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),filters_off])#with eager
        
        #print('\ncounterfactual')
        #print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')
        #print( 'alter class: ',label_map[alter_class], '  prob: ',alter_probs[0][alter_class].numpy()*100,'%')
        #print('alter_pre_softmax: ',alter_pre_softmax[0][0:10])
       
        #modified_model, modified_fmaps = check_histogram_top_filter_result(model,filters_off,x_batch_test[img_ind],y_gt,label_map,args)
        #plt.plot(fmatrix[0]), plt.title('fmatrix'), plt.show()
       
        #modified_mean_fmap_activations = GlobalAveragePooling2D()(modified_fmaps)
        #plt.plot(modified_mean_fmap_activations[0]), plt.title('modified_mean_fmap_activations'), plt.show()
        
        
        gradcam=False
        if gradcam:
            #x_batch_test_nopreprocess
            #x_batch_test
            
            output_cf,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,alter_class,image_nopreprocessed=np.expand_dims(image_nopreprocessed,0),fmatrix=fmatrix,image_weight=0.7)#np.argmin(y_batch_test[img_ind])
            
            plt.imshow(output_cf), plt.axis('off'), plt.title('modified prediction')
            plt.show()
            plt.imshow(restore_original_image_from_array(x_batch_test[img_ind].squeeze())), plt.axis('off'), plt.title('original image')
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

            print('\nthresholded counterfactual')
            print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')
            print( 'alter class: ',label_map[alter_class], '  prob: ',alter_probs[0][alter_class].numpy()*100,'%')
           # print('alter_pre_softmax: ',alter_pre_softmax[0][0:10])
            # if top_k_preds:
            #     k=3
            #     ind = np.argpartition(alter_probs[0], -k)[-k:]
            #     ind = ind[np.argsort(alter_probs[0].numpy()[ind])]
                
            #     for i in range(k):
            #         print('top ',str(i+1)+' predicted: ',label_map[ind[k-1-i]], ' with prob: ',alter_probs[0][ind[k-1-i]].numpy()*100,'%')
           
       
            plt.plot(modified_mean_fmap_activations[0]),plt.ylim([0, np.max(mean_fmap)+1]), plt.title('thresh_mean_fmap_activations'), plt.show()
            
            output_cf,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,alter_class,image_nopreprocessed=np.expand_dims(image_nopreprocessed,0),fmatrix=fmatrix,image_weight=0.7)#np.argmin(y_batch_test[img_ind])
            
            plt.imshow(output_cf), plt.axis('off'), plt.title('thresh prediction')
            plt.show()
            plt.imshow(image_nopreprocessed), plt.axis('off'), plt.title('original image')
            plt.show()
       
        
            fig, axs = plt.subplots(2, 2,figsize=(15,10))
            axs[0, 0].imshow(output_orig), axs[0, 0].axis('off'), axs[0, 0].set_title('original prediction')
            axs[0, 1].imshow(output_cf), axs[0, 1].axis('off'), axs[0, 1].set_title('modified prediction')
            axs[1, 0].plot(mean_fmap[0]),axs[1,0].set_ylim([0, np.max(mean_fmap)+1]), axs[1, 0].set_title('mean_fmaps')
            if args.counterfactual_PP:
                axs[1, 1].plot(modified_mean_fmap_activations[0]),axs[1,1].set_ylim([0, np.max(mean_fmap)+1]), axs[1, 1].set_title('modified_mean_fmap_activations')
            else:
                axs[1, 1].plot(PN_add[0],color='red'),axs[1,1].set_ylim([0, np.max(mean_fmap)+1]), axs[1, 1].set_title('PN_additions')
                axs[1, 1].plot(mean_fmap[0]) ,axs[1,1].set_ylim([0, np.max(mean_fmap)+1]), plt.show()
                
            plt.show()
            
            if not args.counterfactual_PP:
                plt.plot(PN_add[0],color='red')
                plt.plot(mean_fmap[0]), plt.title('PN_additions'),plt.ylim([0, np.max(mean_fmap)+1]), plt.show()
        
        #%% disabled PP prediction:
        # enabled_filters = 1- fmatrix[0]
        # dis_alter_probs, dis_fmaps, dis_mean_fmap, dis_modified_mean_fmap_activations,dis_alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),enabled_filters])#with eager
             
        # print('\nDisabled PP prediction')
        # print( 'pred class: ',label_map[np.argmax(dis_alter_probs)], '  prob: ',dis_alter_probs[0][np.argmax(dis_alter_probs)].numpy()*100,'%')
        # print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',dis_alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')
        # print( 'inferred class: ',label_map[inferred_class], '  prob: ',dis_alter_probs[0][inferred_class].numpy()*100,'%')

        #%% model debugging misclassification analysis
        #compare MC filters of inferred class with the global MC filters of inferred and top-3 classes to find agreement with the inferred class or other classes
        if args.dataset=='CUB200':
            inferred_class = np.argmax(pred_probs)
            
            selected_probs = pred_probs # alter_probs #pred_probs
            top_3_candidate_classes= []
            k=3
            ind = np.argpartition(selected_probs[0], -k)[-k:]
            ind = ind[np.argsort(selected_probs[0].numpy()[ind])]                
            for i in range(k):
                # print('top ',str(i+1)+' predicted: ',label_map[ind[k-1-i]], ' with prob: ',pred_probs[0][ind[k-1-i]].numpy()*100,'%')
                top_3_candidate_classes.append(ind[k-1-i])
            
            scores = []
            #top_3_candidate_classes.append(np.argmax(y_gt))

            #top_3_candidate_classes=[9,25,125,108,170]
            for cand_class in top_3_candidate_classes:#range(200):#top_3_candidate_classes:
                #find agreement with global MC of each class
                #%%
                if len(top_3_candidate_classes)>1:
                    args.alter_class = cand_class
                    alter_class = cand_class
                    if loaded_cfe_model==cand_class:
                        pass
                    else:
                        combined = load_cfe_model()
                        loaded_cfe_model=cand_class
    
                    alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(np.expand_dims(x_batch_test[img_ind],0))
                #%%
                common_filters_count = find_agreement_global_MC(pred_MC=modified_mean_fmap_activations[0], target_class=cand_class, args=args,class_name =label_map[cand_class],class_weights=W[:,cand_class] )
                scores.append(common_filters_count)
            
            print('scores', scores)
            print('top_3_candidate_classes', top_3_candidate_classes)
            print("np.argmax(scores) ", np.argmax(scores))
            print(top_3_candidate_classes[np.argmax(scores)])
        print("")
        continue
end = time.time()
print("time taken: ",end - start)