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


def filter_visualization_top_k(model,gen,T, top_k, args,show_images,gradCAM,class_specific_top_k, RF,img_number=None):
    print("\n\nfinding top k images for target filter ", str(T))

    if gradCAM:
        explainer = GradCAM()

    batches=math.ceil(gen.n/gen.batch_size)
    gen.reset() #resets batch index to 0
    gen.batch_index=0
    
    mean_fmaps_all=None
    
    fileName = args.model[:-1] + '_mean_fmaps_all_'+args.dataset+'.npy'
    if os.path.isfile(fileName):
        mean_fmaps_all = np.load(fileName)
    else:
        for k in range(batches):
            x_batch_test,y_batch_test = next(gen)
            
            sys.stdout.write("\rbatch %i of %i" % (k, batches))
            sys.stdout.flush()
            
            if args.counterfactual_PP:
                default_fmatrix = tf.ones((x_batch_test.shape[0],model.output[1].shape[3]))#512=generator.output.shape[1]
            else:
                default_fmatrix = tf.zeros((x_batch_test.shape[0],model.output[1].shape[3]))#512=generator.output.shape[1]
    
                
            pred_probs,fmaps,mean_fmaps,_ ,pre_softmax= model([x_batch_test,default_fmatrix], training=False)#with eager
            
            
            if (mean_fmaps_all is not None):
                mean_fmaps_all = np.append(mean_fmaps_all,mean_fmaps,axis=0)
            else:
                mean_fmaps_all = mean_fmaps
            # if k==10:
            #     break
        
        mean_fmaps_all = np.asarray(mean_fmaps_all)
        np.save(fileName,mean_fmaps_all)
    
    #####################
    #for global top_k activated images    
    if not class_specific_top_k:
        #sort and keep track of top and bottom k activations for target filter T
        target_fmaps = mean_fmaps_all[:,T]
    
        sort_index = np.argsort(target_fmaps, axis=-1)
        sort_values = target_fmaps[sort_index]
        
        lowest = sort_index[0:top_k]
        highest = sort_index[-top_k:][::-1]#get last k and then reverse order so that best is at index 0
    else:
        #for class-specific top_k activated images    
        
        #class indices (make sure generator is not shuffled?)
        if args.model[:-1]=='myCNN':
            ind = np.where(np.argmax(gen.y,1)==args.alter_class)[0]
        else:
            ind = np.where(gen.classes==args.alter_class)[0]
        #sort and keep track of top and bottom k activations for target filter T
        target_fmaps=[]
        for i in ind:
            target_fmaps.append(mean_fmaps_all[i,T])
        target_fmaps = np.asarray(target_fmaps)
    
        sort_index = np.argsort(target_fmaps, axis=-1)
        sort_values = target_fmaps[sort_index]
        
        lowest = sort_index[0:top_k]
        highest = sort_index[-top_k:][::-1]#get last k
        
        #map the indinces in sub array to actual full array? or directly load these images?
        lowest = ind[lowest]
        highest = ind[highest]
        
        # also reset the target_fmaps to full list to enable rest of the code without changing
        target_fmaps = mean_fmaps_all[:,T]
        
    ####################
    
    #find corresponding images and save their copy
    gen.reset() #resets batch index to 0
    gen.batch_index=0

    ind_low=0
    ind_high=0
    save_dir='./filter_visualization/'+args.model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
     
    fig, axs = plt.subplots(1, top_k,figsize=(15,3.2)) # figsize: (default: [6.4, 4.8]) Width, height in inches.
    fig2, axs2 = plt.subplots(1, top_k,figsize=(15,3.2))
    
    for k in range(batches):
        x_batch_test,y_batch_test = next(gen)
        
        sys.stdout.write("\rbatch %i of %i" % (k, batches))
        sys.stdout.flush()
        
        for i in range (len(x_batch_test)):
            if args.counterfactual_PP:
                #enable just the T filter 
                default_fmatrix = np.zeros((x_batch_test.shape[0],model.output[1].shape[3]))#512=generator.output.shape[1]
                default_fmatrix[i][T] = 1.0
            else:
                assert(args.counterfactual_PP==True)
                #for PN we still only want to see th RF of 1 filter. So check PN filters in PP mode. Since we are not disabling the filters in PN mode.
                
            img_ind = i#3
            
            if gen.batch_index==0:  #for final batch
                actual_img_ind = i + (batches-1)*gen.batch_size
            else:
                actual_img_ind = i + (gen.batch_index-1)*gen.batch_size
            
            if actual_img_ind in lowest:
                #save image lowest
                if args.model[:-1]=='myCNN':# dont preporcess for MNIST on myCNN
                    img_post_process = x_batch_test[i]
                else:
                    img_post_process = restore_original_image_from_array(x_batch_test[i].copy())
                    img_post_process=img_post_process.astype('uint8')
                
                
                img_png = tf.keras.preprocessing.image.array_to_img(img_post_process)
                
                #in case of MNIST
                if img_post_process.shape[2] == 1:
                    img_post_process = img_post_process.squeeze()
                    
                if show_images: plt.imshow(img_post_process),plt.show()
                
                order=np.where(actual_img_ind==lowest)[0][0]
                save_name = save_dir+'filter_'+str(T)+'_lowest_'+str(order)+'_'+str(target_fmaps[lowest[order]])+'.png'
                img_png.save(save_name)
                
                axs[order].imshow(img_post_process), axs[order].axis('off'), axs[order].set_title(str(target_fmaps[lowest[order]]))
                
                ind_low+=1
                #img_png.show()
                if ind_low == top_k and ind_high==top_k: 
                    break
    
            if actual_img_ind in highest:
                #save image highest
                if args.model[:-1]=='myCNN':# dont preporcess for MNIST on myCNN
                    img_post_process = x_batch_test[i]
                else:
                    img_post_process = restore_original_image_from_array(x_batch_test[i].copy())
                    img_post_process=img_post_process.astype('uint8')
                
                
                img_png = tf.keras.preprocessing.image.array_to_img(img_post_process)

                #in case of MNIST
                if img_post_process.shape[2] == 1:
                    img_post_process = img_post_process.squeeze()
                    #resize to 224x224
                    img_post_process = resize(img_post_process, (224, 224))
                    
                
                if show_images: plt.imshow(img_post_process),plt.show()
                
                order=np.where(actual_img_ind==highest)[0][0]
                if order ==3:
                    #print("heatmap   zero debugg")
                    pass
                save_name = save_dir+'filter_'+str(T)+'_highest_'+str(order)+'_'+str(target_fmaps[highest[order]])+'.png'
                img_png.save(save_name)
                
                alter_class = True #show heatmap for alter class or actual predicted class
                #keep all filter on to find the target class, but not for genereating heatmap
                if not alter_class:
                    all_ones = np.ones(model.output[1].shape[3])#512=generator.output.shape[1]
                    pred_probs,fmaps,mean_fmaps,_ ,pre_softmax= model([np.expand_dims(x_batch_test[i],0),np.expand_dims(all_ones,0)], training=False)#with eager
                    target_class = np.argmax(pred_probs)
                else:
                    target_class = args.alter_class

                if gradCAM:
                    output_gradcam_alter,_ = explainer.explain((np.expand_dims(x_batch_test[i],0),None),model,target_class,image_nopreprocessed=np.expand_dims(img_post_process,0),fmatrix=np.expand_dims(default_fmatrix[i],0),image_weight=0.7, RF=RF)
                    axs2[order].imshow(output_gradcam_alter), axs2[order].axis('off'), axs2[order].set_title(str(target_fmaps[highest[order]]))
                else:
                    axs2[order].imshow(img_post_process), axs2[order].axis('off'), axs2[order].set_title(str(target_fmaps[highest[order]]))

                    
                ind_high+=1
                
                if ind_low == top_k and ind_high==top_k: 
                    break
        if ind_low == top_k and ind_high==top_k: 
            break
    if class_specific_top_k:
        bot_k_title = "Bottom 5 class-specific images for filter "+ str(T)
        top_k_title = "Top 5 class-specific images for filter "+ str(T)        
    else: 
        bot_k_title = "Bottom 5 images for filter "+ str(T)
        top_k_title = "Top 5 images for filter "+ str(T)   
    
    fig.suptitle(bot_k_title)
    fig2.suptitle(top_k_title)
    
    if args.user_evaluation:
        plt.savefig(fname="./Comparison with SCOUT/"+str(img_number)+"_PN_filter_"+str(T)+".png", dpi=300, bbox_inches = 'tight')

    plt.show()
    
    #TODO: plot in sorted order

        


    
    