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

#%%
from config import args, weights_path, KAGGLE, pretrained_weights_path
from load_data import top_activation, num_classes, train_gen, label_map, actual_test_gen
from load_base_model import base_model
from load_CFE_model import model, load_cfe_model, fmatrix
from codes.filter_visualization_same_image import filter_visualization_same_image

#%%
np.random.seed(seed=100)


    
assert(args.find_global_filters==False) #to make sure data generators are setup properly
assert(args.counterfactual_PP)
assert(args.train_all_classes) # to make sure CFE model is loaded from correct path

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



class_for_analysis = args.analysis_class#args.analysis_class#9#9 170#np.random.randint(200)#23#11 #cat for VOC dataset
#alter_class=args.alter_class
print ('class for analysis: ', label_map[class_for_analysis])
#print ('alter class: ', label_map[alter_class])

weights_alter_class = W[:,args.analysis_class]
plt.plot(weights_alter_class),plt.title("weight alter class "+str(args.analysis_class)),plt.show()

if args.dataset == 'CUB200' or args.dataset == 'BraTS' or args.dataset == 'NIST': 
    test_gen =train_gen if args.find_global_filters else actual_test_gen# train_gen#actual_test_gen
    #test_gen_nopreprocess = train_gen_nopreprocess if args.find_global_filters else actual_test_gen_nopreprocess #train_gen_nopreprocess[0]#actual_test_gen_nopreprocess
    # print("using traingen gen data") if args.find_global_filters else print("using testgen gen data")

gen=test_gen
batches=math.ceil(gen.n/gen.batch_size)



filter_histogram_cf = tf.zeros(fmatrix[0].shape[0])
filter_magnitude_cf = tf.zeros(fmatrix[0].shape[0])
filter_sum = 0

def get_cfe_MC_prediction(alter_class):
    combined = load_cfe_model(alter_class)
    alter_probs,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(np.expand_dims(x_batch_test[img_ind],0))
    print('\nthresholded counterfactual')
    print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')
    print( 'alter class: ',label_map[alter_class], '  prob: ',alter_probs[0][alter_class].numpy()*100,'%')
    
    top_k=3
    sort_by_weighted_magnitude = True
    # if sort_by_weighted_magnitude:
    weighted_activation_magnitudes = modified_mean_fmap_activations[0]*W[:,alter_class]
    top_3_filters_weigted_mag = np.argsort(weighted_activation_magnitudes)[-top_k:][::-1]#top 3
    # else:
    top_3_filters_mag_alone = np.argsort(modified_mean_fmap_activations[0])[-top_k:][::-1]

    filter_visualization_same_image(model,combined, x_batch_test[img_ind],actual_img_ind,top_3_filters_weigted_mag, args,show_images=False, gradCAM=True, RF = True, combined_heatmaps = False)
    filter_visualization_same_image(model,combined, x_batch_test[img_ind],actual_img_ind,top_3_filters_mag_alone, args,show_images=False, gradCAM=True, RF = True, combined_heatmaps = False)

    return alter_probs,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax
    
    
def get_original_predictions(show_fig):
    model.load_weights(filepath=pretrained_weights_path+'/model_fine_tune_epoch_150.hdf5')
    pred_probs,fmaps,mean_fmaps,_ ,pre_softmax= model([np.expand_dims(x_batch_test[img_ind],0),np.expand_dims(default_fmatrix[0],0)], training=False)#with eager
    
    output_orig2=[]
    if show_fig:
        print('\noriginal predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
        print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%')

        image_nopreprocessed = restore_original_image_from_array(x_batch_test[img_ind].squeeze())
        output_orig,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,np.argmax(pred_probs),image_nopreprocessed=np.expand_dims(image_nopreprocessed,0),fmatrix=default_fmatrix,image_weight=0.7, RF=False, heatmap_cutout = False)
        plt.imshow(output_orig), plt.axis('off'), plt.title('original predicted')
        plt.show()
        output_orig2,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,np.argmax(pred_probs),image_nopreprocessed=np.expand_dims(image_nopreprocessed,0),fmatrix=default_fmatrix,image_weight=0.7, RF=True, heatmap_cutout = False)
        plt.imshow(output_orig2), plt.axis('off'), plt.title('original predicted')
        plt.show()
        
    #cf MC prediction
    #Todo: cannot use CFE model for debugging analysis because all weights saved for CFE model are based on original base model. no the improved model. 
    #So I think it is best to use gradCAMs to show variation in model attention in improved models.
        # alter_class=np.argmax(pred_probs)
        # alter_probs,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,_ = get_cfe_MC_prediction(alter_class)
    
    return pred_probs, output_orig2

def get_debugged_predictions(show_fig):
    model.load_weights(filepath=pretrained_weights_path+'/model_debugged_0.7033407_7064.hdf5')
    pred_probs,fmaps,mean_fmaps,_ ,pre_softmax= model([np.expand_dims(x_batch_test[img_ind],0),np.expand_dims(default_fmatrix[0],0)], training=False)#with eager
    
    output_debugged2=[]
    if show_fig:
        print('\ndebugged predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
        print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%')

        image_nopreprocessed = restore_original_image_from_array(x_batch_test[img_ind].squeeze())
        output_debugged,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,np.argmax(pred_probs),image_nopreprocessed=np.expand_dims(image_nopreprocessed,0),fmatrix=default_fmatrix,image_weight=0.7, RF=False, heatmap_cutout = False)
        plt.imshow(output_debugged), plt.axis('off'), plt.title('debugged predicted')
        plt.show()
        output_debugged2,_ = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,np.argmax(pred_probs),image_nopreprocessed=np.expand_dims(image_nopreprocessed,0),fmatrix=default_fmatrix,image_weight=0.7, RF=True, heatmap_cutout = False)
        plt.imshow(output_debugged2), plt.axis('off'), plt.title('debugged predicted')
        plt.show()
    #cf MC prediction
    #Todo: cannot use CFE model for debugging analysis because all weights saved for CFE model are based on original base model. no the improved model. 
    #So I think it is best to use gradCAMs to show variation in model attention in improved models.
        # alter_class=np.argmax(pred_probs)
        # alter_probs,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,_ = get_cfe_MC_prediction(alter_class)
    

    return pred_probs, output_debugged2           

gen.reset() #resets batch index to 0
local_misclassifications = 0
img_count=0
alter_class_images_count = np.sum(gen.classes==args.analysis_class)
alter_class_starting_batch = np.floor(np.where(gen.labels==args.analysis_class)[0][0]/gen.batch_size).astype(np.int32)
index_reached=0
loaded_cfe_model=-1
start = time.time()
skip = False# to skip to chosen class images
for k in range(batches):

    if k < alter_class_starting_batch and skip:
        sys.stdout.write("\rskipping batch %i of %i" % (k, batches))
        continue
    else:
        x_batch_test,y_batch_test = next(gen)                
        gen.batch_index = k

        
    x_batch_test,y_batch_test = next(gen)
    
    default_fmatrix = tf.ones((len(x_batch_test),base_model.output.shape[3]))

    if gen.batch_index < alter_class_starting_batch and gen.batch_index >0 and skip:
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
        if class_for_analysis==np.argmax(y_gt) or not skip:
            pass
            # print('\n\nimg_ind:',actual_img_ind)
        else:
            continue
        
        print("\n------------------------------------------")
        #original model
        pred_probs_original,output_orig = get_original_predictions(show_fig = False)
        #debugged model
        pred_probs_debugged,output_debugged = get_debugged_predictions(show_fig = False)
        
        case = ""
        if np.argmax(pred_probs_original) != np.argmax(y_gt) and np.argmax(pred_probs_debugged) == np.argmax(y_gt): 
            case='Misclassification corrected'
            print("\n",case)
        if np.argmax(pred_probs_original) == np.argmax(y_gt) and np.argmax(pred_probs_debugged) != np.argmax(y_gt):
            case = "New Misclassification"
            print("\n",case)
        if np.argmax(pred_probs_original) == np.argmax(y_gt) and np.argmax(pred_probs_debugged) == np.argmax(y_gt): 
            if np.max(pred_probs_debugged)> np.max(pred_probs_original) and (np.max(pred_probs_debugged)-np.max(pred_probs_original))>0.3:
                case = "Confidence improved"
                print("\n",case)
        if np.argmax(pred_probs_original) != np.argmax(y_gt) and np.argmax(pred_probs_debugged) != np.argmax(y_gt): 
            print("\nMisclassification NOT corrected")

        print("\n------------------------------------------")
        
        skip_high_confidence = False
        if skip_high_confidence:
            if pred_probs_original[0][np.argmax(y_gt)]>0.9:
                print("skipping high confidence prediction")
                continue
        skip_low_confidence_debugged = False
        if skip_low_confidence_debugged:
            if pred_probs_debugged[0][np.argmax(y_gt)]<0.95:
                print("skipping low confidence prediction")
                continue


        save_fig=False
        # if case!='':
        if case=='Confidence improved':
            pred_probs_original,output_orig = get_original_predictions(show_fig = True)
            #debugged model
            pred_probs_debugged,output_debugged = get_debugged_predictions(show_fig = True)

            # save_fig = int(input("save fig?"))
            if save_fig or True:
                mName = args.model[:-1]+'_'+args.dataset
                
                # plt.imshow(output_orig), plt.axis('off'), plt.title(''),plt.savefig(fname="./model_debugging_work/figures/"+case+'/'+mName+"_orig_GradCAM_"+str(np.argmax(pred_probs_original))+"_"+str(np.max(pred_probs_original))+"_"+str(actual_img_ind)+".png", dpi=300, bbox_inches = 'tight'), plt.show()
                # plt.imshow(output_debugged), plt.axis('off'), plt.title(''),plt.savefig(fname="./model_debugging_work/figures/"+case+'/'+mName+"_debuggedGradCAM_alter_"+str(np.argmax(pred_probs_debugged))+"_"+str(np.max(pred_probs_debugged))+"_"+str(actual_img_ind)+".png", dpi=300, bbox_inches = 'tight'), plt.show()
                plt.imshow(output_orig), plt.axis('off'), plt.title(''),plt.savefig(fname="./model_debugging_work/figures/"+case+'/'+mName+"_"+str(actual_img_ind)+"_orig_GradCAM_"+str(np.argmax(pred_probs_original))+"_"+str(np.max(pred_probs_original))+".png", dpi=300, bbox_inches = 'tight'), plt.show()
                plt.imshow(output_debugged), plt.axis('off'), plt.title(''),plt.savefig(fname="./model_debugging_work/figures/"+case+'/'+mName+"_"+str(actual_img_ind)+"_debuggedGradCAM_alter_"+str(np.argmax(pred_probs_debugged))+"_"+str(np.max(pred_probs_debugged))+".png", dpi=300, bbox_inches = 'tight'), plt.show()


        # continue
  
        #%%
            


end = time.time()
print("time taken: ",end - start)