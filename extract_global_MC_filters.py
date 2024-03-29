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
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import math
import time
from tf_explain_modified.core.grad_cam import GradCAM
import warnings
#%%
from codes.support_functions import restore_original_image_from_array
from codes.find_agreement_global_MC import find_agreement_global_MC

#%%
from config import args, weights_path, KAGGLE, pretrained_weights_path
from load_data import top_activation, num_classes, train_gen, label_map
from load_base_model import base_model
from load_CFE_model import model, load_cfe_model, fmatrix

#%%
np.random.seed(seed=100)


    
assert(args.find_global_filters) #to make sure data generators are setup properly
assert(args.counterfactual_PP)
assert(args.train_all_classes)

#%% create base model

W = model.weights[-2]

#%%
explainer = GradCAM()

wrong_inferred_predictions = 0
wrong_CFE_MC_inferred_predictions = 0
low_confidence_predictions = 0
total_images_in_MC_global_filters=0
disabled_MC_filters=0
start = time.time()
enable_prints=False

all_classes=True
if all_classes:
    print("checking all_classes")
    selected_classes = range(200)
else:
    print("checking selected_classes")    
    selected_classes = [9,25,108,125,170]

for loop in selected_classes:#range(num_classes):#range(1):#num_classes):
    args.alter_class = loop#9#loop
    metrics_aggregate = []
    combined = load_cfe_model()

    class_for_analysis = args.alter_class#args.analysis_class#9#9 170#np.random.randint(200)#23#11 #cat for VOC dataset
    alter_class=args.alter_class
    # print ('class for analysis: ', label_map[class_for_analysis])
    print ('\nalter class: ', label_map[alter_class])
    #print ('class 2: ', label_map[args.alter_class_2])
    
   
    if args.dataset == 'CUB200' or args.dataset == 'BraTS' or args.dataset == 'NIST': 
        test_gen =train_gen #if args.find_global_filters else actual_test_gen# train_gen#actual_test_gen
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
    alter_class_images_count = np.sum(gen.classes==alter_class)
    alter_class_starting_batch = np.floor(np.where(gen.labels==args.alter_class)[0][0]/gen.batch_size).astype(np.int32)
    index_reached=0
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
            
                    

            y_gt = y_batch_test[img_ind]
            
             #skip other class        
            if class_for_analysis==np.argmax(y_gt):
                pass
                # print('\n\nimg_ind:',actual_img_ind)
            else:
                continue
            
            #%%
            #compute histpgram of activated filters
            #keep track of activation magnitude

            if True:#sum(y_batch_test[:,args.alter_class])<len(x_batch_test):
                #process sequentially
                
                pred_probs,fmaps,mean_fmaps,_ ,pre_softmax= model([np.expand_dims(x_batch_test[img_ind],0),np.expand_dims(default_fmatrix[0],0)], training=False)#with eager
                # print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
                # print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%')
                
                
                gradcam = False
                if gradcam:
                    image_nopreprocessed = restore_original_image_from_array(x_batch_test[img_ind].squeeze())
                    output_orig,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,np.argmax(pred_probs),image_nopreprocessed=np.expand_dims(image_nopreprocessed,0),fmatrix=default_fmatrix)
                    
                    plt.imshow(output_orig), plt.axis('off'), plt.title('original prediction')
                    plt.show()
                                 

                if np.argmax(pred_probs) != np.argmax(y_gt) and True:
                    if enable_prints: print("wrong prediction")
                    # incorrect_class=np.argmax(pred_probs)
                    if enable_prints:print("skipping wrong prediction")
                    filter_sum += 1
                    wrong_inferred_predictions+=1
                    continue
                else:
                    pass
                    # print("skipping correct prediction")
                    # continue
                
               # #skip low confidence predictions
                skip_low_confidence = True
                if skip_low_confidence:
                    if pred_probs[0][np.argmax(y_gt)]<0.9:
                        if enable_prints: print("skipping low confidence prediction")
                        filter_sum += 1
                        low_confidence_predictions+=1
                        continue
 
                    
                alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(np.expand_dims(x_batch_test[img_ind],0))
          
                if np.argmax(alter_prediction) != np.argmax(y_gt) and True:
                    if enable_prints: print("wrong CFE model prediction")
                    # incorrect_class=np.argmax(pred_probs)
                    if enable_prints: print("skipping wrong MC CFE model prediction")
                    filter_sum += 1
                    wrong_CFE_MC_inferred_predictions+=1
                    
                    continue
                else:
                    pass
                    # print("skipping correct prediction")
                    # continue
                #%% disabled PP prediction:

                skip_disabled_MC_filters = False
                if skip_disabled_MC_filters:
                    enabled_filters = 1- fmatrix[0]
                    dis_alter_probs, dis_fmaps, dis_mean_fmap, dis_modified_mean_fmap_activations,dis_alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),enabled_filters])#with eager
                         
                    print('\nDisabled PP prediction')
                    print( 'pred class: ',label_map[np.argmax(dis_alter_probs)], '  prob: ',dis_alter_probs[0][np.argmax(dis_alter_probs)].numpy()*100,'%')
                    print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',dis_alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')

                    if np.argmax(dis_alter_probs) == np.argmax(alter_prediction):
                        filter_sum += 1
                        disabled_MC_filters+=1
                        continue
                    

                #%%
                
                filter_histogram_cf += fmatrix[0]
                filter_magnitude_cf += modified_mean_fmap_activations[0]
                filter_sum += 1
                # print("image_count",filter_sum)
                total_images_in_MC_global_filters+=1
                # sys.stdout.write("\rimage_count %i" % (filter_sum))
                # sys.stderr.flush()
                #%%
                warnings.warn("Make sure that global MC filters are previously extracted in order to find average metrics for martching MC filters with global MC filters")
                
                freq_thresh=0.15
                metrics = find_agreement_global_MC(freq_thresh=freq_thresh,pred_MC=modified_mean_fmap_activations[0], target_class=alter_class, args=args,class_name =label_map[alter_class],class_weights=W[:,alter_class] )
                metrics_aggregate.append(metrics)
                #%%
            else:
                assert(False) # following code not updated
                #process in batch
                alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(x_batch_test)                
            
                t_fmatrix = fmatrix.numpy()
                for i in tf.where(fmatrix>0):
                    t_fmatrix[tuple(i)]=1.0
                t_fmatrix = tf.convert_to_tensor(t_fmatrix)
                alter_probs, c_fmaps, c_mean_fmap, c_modified_mean_fmap_activations,alter_pre_softmax = model([x_batch_test,t_fmatrix])#with eager
                
                # print('thresholded counterfactual')
                # print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')
                # print( 'alter class: ',label_map[alter_class], '  prob: ',alter_probs[0][alter_class].numpy()*100,'%')
                # print('alter_pre_softmax: ',alter_pre_softmax[0][0:10])
                
                for i in range(len(t_fmatrix)):
                    filter_histogram_cf += t_fmatrix[i] 
                    filter_magnitude_cf += c_modified_mean_fmap_activations[i]
                
                filter_sum += len(t_fmatrix)
                print("image_count",filter_sum)
                #only break if its not the last batch
                if k == batches-1:
                    pass
                else:
                    break
                # sys.stdout.write("\rimage_count %i" % (filter_sum))
                # sys.stderr.flush()
            
            
            if args.dataset=='mnist':
                test_image_count = [ 980,
                                    1135,
                                    1032,
                                    1010,
                                     982,
                                     892,
                                     958,
                                    1028,
                                     974,
                                    1009]
            else:
                # test_image_count = 30
                test_image_count = alter_class_images_count
            if filter_sum==alter_class_images_count:#test_image_count[class_for_analysis]:
                
                if enable_prints: print("\nfinished")
                # plt.plot(filter_histogram_cf), plt.show()
                # plt.plot(filter_magnitude_cf), plt.show()
               # plt.plot(filter_magnitude_cf/(filter_histogram_cf+0.00001)), plt.show()
                
                mName = args.model[:-1]+'_'+args.dataset
                #plt.ylim([0, np.max(c_mean_fmap)+1]), 
                
                # save_folder = "./figs_for_paper/"
                save_folder = "./model_debugging_work/epoch_"+str(args.cfe_epochs)+"/"+str(label_map[alter_class])+"/"
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                
                # plt.plot(filter_histogram_cf), plt.ylim([0, np.max(filter_histogram_cf)+1]),plt.xlabel("Filter number"),plt.ylabel("Filter activation count"), plt.savefig(fname=save_folder+mName+"_filter_histogram_cf_alter_class_"+str(alter_class)+"_"+str(class_for_analysis)+"_train_set.png", dpi=300, bbox_inches = 'tight'), plt.show()
                # plt.plot(filter_magnitude_cf/max(filter_histogram_cf)),plt.xlabel("Filter number"),plt.ylabel("Avg. activation magnitude"), plt.ylim([0, np.max(filter_magnitude_cf/np.max(filter_histogram_cf))+1]), plt.savefig(fname=save_folder+mName+"_normalized_filter_magnitude_cf_alter_class_"+str(alter_class)+"_"+str(class_for_analysis)+"_train_set.png", dpi=300, bbox_inches = 'tight'), plt.show()
                #plt.plot(filter_magnitude_cf/max(filter_histogram_cf)), plt.ylim([0, np.max(filter_magnitude_cf/max(filter_histogram_cf))+1]), plt.show()
                
                np.save(file= save_folder+mName+"_filter_histogram_cf_"+str(alter_class)+"_train_set.np",arr=filter_histogram_cf)
                np.save(file= save_folder+mName+"_normalized_filter_magnitude_cf_"+str(alter_class)+"_train_set.np", arr=filter_magnitude_cf)
                
                metrics_aggregate=np.asarray(metrics_aggregate)
                np.save(file= save_folder+mName+"_metrics_aggregate_freq_thresh_"+str(freq_thresh)+"_"+str(alter_class)+"_train_set.np", arr=metrics_aggregate)
                
                #######################
                #thresholded stats
                
                
                filter_histogram_cf_thresholded = np.zeros_like(filter_histogram_cf)
                thresh=9
                filter_histogram_cf_thresholded[filter_histogram_cf>thresh] = filter_histogram_cf[filter_histogram_cf>thresh]
                
                #plt.plot(filter_histogram_cf_thresholded), plt.ylim([0, np.max(filter_histogram_cf_thresholded)+1]),plt.savefig(fname="./figs_for_paper/"+mName+"_filter_histogram_thresholded_"+str(thresh)+"_cf_alter_class_"+str(alter_class)+"_"+str(class_for_analysis)+"_test_set.png", dpi=None, bbox_inches = 'tight'), plt.show()
                #TODO::plt.plot(filter_magnitude_cf/max(filter_histogram_cf)), plt.ylim([0, np.max(filter_magnitude_cf/np.max(filter_histogram_cf))+1]), plt.savefig(fname="./figs_for_paper/"+mName+"_normalized_filter_magnitude_thresholded_"+str(thresh)+"_cf_alter_class_"+str(alter_class)+"_"+str(class_for_analysis)+"_train_set.png", dpi=None, bbox_inches = 'tight'), plt.show()
                break
            continue
        #end batch
        if filter_sum==alter_class_images_count: break

end = time.time()
print("wrong_inferred_predictions", wrong_inferred_predictions)
print("wrong_CFE_MC_inferred_predictions", wrong_CFE_MC_inferred_predictions)
print("low_confidence_predictions", low_confidence_predictions)
print("total_images_in_MC_global_filters", total_images_in_MC_global_filters)

print("disabled_MC_filters", disabled_MC_filters)


print("time taken: ",end - start)