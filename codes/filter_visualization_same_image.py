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


def filter_visualization_same_image(model,combined,img,img_ind,filts, args,show_images,gradCAM, RF, combined_heatmaps):
    print("\n\nfinding image heatmpas for target filters ", str(filts))
    
    top_k = len(filts)
    
    if gradCAM:
        explainer = GradCAM()

    save_dir='./image_filter_visualization/'+args.model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
     
    fig, axs = plt.subplots(1, top_k,figsize=(10,3.2)) # figsize: (default: [6.4, 4.8]) Width, height in inches.
    
    
    img_post_process = restore_original_image_from_array(img.copy())
    img_post_process=img_post_process.astype('uint8')
    
            
    img_png = tf.keras.preprocessing.image.array_to_img(img_post_process)
            
    #in case of MNIST
    if img_post_process.shape[2] == 1:
        img_post_process = img_post_process.squeeze()
        
    if show_images: plt.imshow(img_post_process),plt.show()

    #save_name = save_dir+'image_'+str(img_ind)+'_filter_'+str(filts[i])+'.png'
    # img_png.save(save_name)    

    target_class = args.alter_class
    for i in range(top_k):
        if args.counterfactual_PP:
            #enable just the T filter 
            default_fmatrix = np.zeros((1, model.output[1].shape[3]))#512=generator.output.shape[1]
            default_fmatrix[0][filts[i]] = 1.0
        else:
            assert(args.counterfactual_PP==True)
            #for PN we still only want to see th RF of 1 filter. So check PN filters in PP mode. Since we are not disabling the filters in PN mode.
            
            #or set activation fmaps to zero by setting fmatrix to negative of actual fmatrix except for the ref filter 
            alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax,PN_add = combined(np.expand_dims(img,0))#with eager
            default_fmatrix = -mean_fmap.numpy()
            default_fmatrix[0][filts[i]] = fmatrix[0][filts[i]]

            dis_alter_probs, dis_fmaps, dis_mean_fmap, dis_modified_mean_fmap_activations,dis_alter_pre_softmax = model([np.expand_dims(img,0),default_fmatrix])#with eager

    
        order=i
        if gradCAM:
            output_gradcam_alter,_ = explainer.explain((np.expand_dims(img,0),None),model,target_class,image_nopreprocessed=np.expand_dims(img_post_process,0),fmatrix=np.expand_dims(default_fmatrix[0],0),image_weight=0.7, RF=RF)
            axs[order].imshow(output_gradcam_alter), axs[order].axis('off'), axs[order].set_title(str(filts[i]))
            
            if combined_heatmaps: img_post_process = output_gradcam_alter
        else:
            axs[order].imshow(img_post_process), axs[order].axis('off'), axs[order].set_title(str(filts[i]))

                    

    # bot_k_title = "Bottom 5 images for filter "+ str(T)
    # top_k_title = "Top 5 images for filter "+ str(T)   
    
    # fig.suptitle(bot_k_title)
    # fig2.suptitle(top_k_title)
    plt.show()
    
    #TODO: plot in sorted order

        


    
    