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
from tf_explain_modified.core.grad_cam import GradCAM

#%%
from codes.support_functions import restore_original_image_from_array

#%%
from config import args, weights_path, KAGGLE, pretrained_weights_path
from load_data import top_activation, num_classes, train_gen, label_map
from load_base_model import base_model

#%%
np.random.seed(seed=100)


    
assert(args.find_global_filters) #to make sure data generators are setup properly
assert(args.counterfactual_PP)
assert(args.train_all_classes)

#%% create base model
top_filters = base_model.output_shape[3] # flters in top conv layer (512 for VGG)
fmatrix = tf.keras.layers.Input(shape=(top_filters),name='fmatrix')
#flag = tf.keras.layers.Input(shape=(1))

if args.model == 'VGG16/' or args.model == 'myCNN/':
    x =  MaxPool2D()(base_model.output)
elif args.model == 'resnet50/':
    x =  base_model.output
elif args.model == 'efficientnet/':
    x =  base_model.output
mean_fmap = GlobalAveragePooling2D()(x)


#modify base model (once it has been pre-trained separately) to be used with CF model later
if args.counterfactual_PP:
    modified_fmap = mean_fmap*fmatrix
else:#PN
    modified_fmap = mean_fmap+fmatrix
pre_softmax = Dense(num_classes,activation=None)(modified_fmap)
out = tf.keras.layers.Activation(top_activation)(pre_softmax)
model = tf.keras.Model(inputs=[base_model.input, fmatrix], outputs= [out,base_model.output, mean_fmap, modified_fmap,pre_softmax],name='base_model')

if args.counterfactual_PP:
    default_fmatrix = tf.ones((train_gen.batch_size,base_model.output.shape[3]))
else:
    default_fmatrix = tf.zeros((train_gen.batch_size,base_model.output.shape[3]))


#model.summary()

#load saved weights
if args.model =='myCNN/':
    model.load_weights(filepath=pretrained_weights_path+'/model_transfer_epoch_50.hdf5')
else:
    model.load_weights(filepath=pretrained_weights_path+'/model_fine_tune_epoch_150.hdf5')

print("base model weights loaded")

#%%
W = model.weights[-2]
act_threshold=-0.15
plt.plot(W.numpy()), plt.title('Weights'), plt.legend(['cat','dog']),
plt.hlines(act_threshold,xmin=0,xmax=512,colors='r'),plt.show()

important_filter_weights_1 = np.where(W[:,0]<=act_threshold)
important_filter_weights_2 = np.where(W[:,1]<=act_threshold)


#%% create CFE model
num_filters = model.output[1].shape[3]
model.trainable = False

if args.model == 'VGG16/' or args.model == 'myCNN/':
    x =  MaxPool2D()(base_model.output)
elif args.model == 'resnet50/':
    x =  base_model.output
elif args.model == 'efficientnet/':
    x =  base_model.output
mean_fmap = GlobalAveragePooling2D()(x)

if args.counterfactual_PP:
    x = Dense(num_filters,activation='sigmoid')(mean_fmap)#kernel_regularizer='l1' #,activity_regularizer='l1'
else:
    x = Dense(num_filters,activation='relu')(mean_fmap)


thresh=0.5
PP_filter_matrix = tf.keras.layers.ThresholdedReLU(theta=thresh)(x)



counterfactual_generator = tf.keras.Model(inputs=base_model.input, outputs= [PP_filter_matrix],name='counterfactual_model')


def load_cfe_model():
    if not args.train_singular_counterfactual_net:
        if args.choose_subclass:
            counterfactual_generator.load_weights(filepath=weights_path+'/counterfactual_generator_model_only_010.Red_winged_Blackbird_alter_class_epochs_'+str(args.cfe_epochs)+'.hdf5')
        else:                
            if args.counterfactual_PP:
                mode = '' 
                print("Loading CF model for PPs")
            else:
                mode = 'PN_'
                print("Loading CF model for PNs")
            counterfactual_generator.load_weights(filepath=weights_path+'/'+mode+'counterfactual_generator_model_fixed_'+str(label_map[args.alter_class])+'_alter_class_epochs_'+str(args.cfe_epochs)+'.hdf5')
    else:
        counterfactual_generator.load_weights(filepath=weights_path+'/counterfactual_generator_model_ALL_classes_epoch_131.hdf5')
        
    model.trainable = False
    img = tf.keras.Input(shape=model.input_shape[0][1:4])

    fmatrix = counterfactual_generator(img)

    alter_prediction,fmaps,mean_fmap, modified_mean_fmap_activations,pre_softmax = model([img,fmatrix])
    
    combined = tf.keras.Model(inputs=img, outputs=[alter_prediction,fmatrix,fmaps,mean_fmap,modified_mean_fmap_activations,pre_softmax])
    return combined

    
#%%
explainer = GradCAM()
misclassification_analysis = False

wrong_inferred_predictions = 0
wrong_CFE_MC_inferred_predictions = 0
low_confidence_predictions = 0
total_images_in_MC_global_filters=0
for loop in range(num_classes):#range(1):#num_classes):
    args.alter_class = loop#9#loop

    combined = load_cfe_model()

    class_for_analysis = args.alter_class#args.analysis_class#9#9 170#np.random.randint(200)#23#11 #cat for VOC dataset
    alter_class=args.alter_class
    # print ('class for analysis: ', label_map[class_for_analysis])
    print ('alter class: ', label_map[alter_class])
    #print ('class 2: ', label_map[args.alter_class_2])
    
    weights_alter_class = W[:,alter_class]
    plt.plot(weights_alter_class),plt.title("weight alter class "+str(alter_class)),plt.show()
    
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
                    print("wrong prediction")
                    # incorrect_class=np.argmax(pred_probs)
                    print("skipping wrong prediction")
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
                        print("skipping low confidence prediction")
                        filter_sum += 1
                        low_confidence_predictions+=1
                        continue
 
                    
                alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(np.expand_dims(x_batch_test[img_ind],0))
                filters_off = fmatrix
          
                apply_thresh = True
                if apply_thresh:
                    t_fmatrix = filters_off.numpy()
                    if args.counterfactual_PP:
                        for i in tf.where(filters_off>0):
                            t_fmatrix[tuple(i)]=1.0
                        t_fmatrix = tf.convert_to_tensor(t_fmatrix)
                    alter_probs, c_fmaps, c_mean_fmap, c_modified_mean_fmap_activations,alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),t_fmatrix])#with eager
                    
                    # print('\nthresholded counterfactual')
                    # print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',alter_probs[0][np.argmax(y_gt)].numpy()*100,'%')
                    # print( 'alter class: ',label_map[alter_class], '  prob: ',alter_probs[0][alter_class].numpy()*100,'%')


                if np.argmax(alter_probs) != np.argmax(y_gt) and True:
                    print("wrong CFE model prediction")
                    # incorrect_class=np.argmax(pred_probs)
                    print("skipping wrong MC CFE model prediction")
                    filter_sum += 1
                    wrong_CFE_MC_inferred_predictions+=1
                    
                    continue
                else:
                    pass
                    # print("skipping correct prediction")
                    # continue
                
                ###############################
                #previous code
                # alter_prediction,fmatrix,fmaps, mean_fmap, modified_mean_fmap_activations,alter_pre_softmax = combined(np.expand_dims(x_batch_test[img_ind],0))                
            
                # t_fmatrix = fmatrix.numpy()
                # for i in tf.where(fmatrix>0):
                #     t_fmatrix[i]=1.0
                # t_fmatrix = tf.convert_to_tensor(t_fmatrix)
                # alter_probs, c_fmaps, c_mean_fmap, c_modified_mean_fmap_activations,alter_pre_softmax = model([np.expand_dims(x_batch_test[img_ind],0),t_fmatrix])#with eager
                ###############################

                
                filter_histogram_cf += t_fmatrix[0]
                filter_magnitude_cf += c_modified_mean_fmap_activations[0]
                filter_sum += 1
                # print("image_count",filter_sum)
                total_images_in_MC_global_filters+=1
                # sys.stdout.write("\rimage_count %i" % (filter_sum))
                # sys.stderr.flush()
            else:
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
                
                print("\nfinished")
                # plt.plot(filter_histogram_cf), plt.show()
                # plt.plot(filter_magnitude_cf), plt.show()
               # plt.plot(filter_magnitude_cf/(filter_histogram_cf+0.00001)), plt.show()
                
                mName = args.model[:-1]+'_'+args.dataset
                #plt.ylim([0, np.max(c_mean_fmap)+1]), 
                
                # save_folder = "./figs_for_paper/"
                save_folder = "./model_debugging_work/epoch_"+str(args.cfe_epochs)+"/"+str(label_map[alter_class])+"/"
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                
                plt.plot(filter_histogram_cf), plt.ylim([0, np.max(filter_histogram_cf)+1]),plt.xlabel("Filter number"),plt.ylabel("Filter activation count"), plt.savefig(fname=save_folder+mName+"_filter_histogram_cf_alter_class_"+str(alter_class)+"_"+str(class_for_analysis)+"_train_set.png", dpi=300, bbox_inches = 'tight'), plt.show()
                plt.plot(filter_magnitude_cf/max(filter_histogram_cf)),plt.xlabel("Filter number"),plt.ylabel("Avg. activation magnitude"), plt.ylim([0, np.max(filter_magnitude_cf/np.max(filter_histogram_cf))+1]), plt.savefig(fname=save_folder+mName+"_normalized_filter_magnitude_cf_alter_class_"+str(alter_class)+"_"+str(class_for_analysis)+"_train_set.png", dpi=300, bbox_inches = 'tight'), plt.show()
                #plt.plot(filter_magnitude_cf/max(filter_histogram_cf)), plt.ylim([0, np.max(filter_magnitude_cf/max(filter_histogram_cf))+1]), plt.show()
                
                np.save(file= save_folder+mName+"_filter_histogram_cf_"+str(alter_class)+"_train_set.np",arr=filter_histogram_cf)
                np.save(file= save_folder+mName+"_normalized_filter_magnitude_cf_"+str(alter_class)+"_train_set.np", arr=filter_magnitude_cf)
                
                
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

print("wrong_inferred_predictions", wrong_inferred_predictions)
print("wrong_CFE_MC_inferred_predictions", wrong_CFE_MC_inferred_predictions)
print("low_confidence_predictions", low_confidence_predictions)
print("total_images_in_MC_global_filters", total_images_in_MC_global_filters)
