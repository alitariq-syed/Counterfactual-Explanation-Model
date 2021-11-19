# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:59:31 2020

@author: Ali
"""
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

#%%
def get_class_map(filters,num_classes):
      k=filters
      n_classes = num_classes
      
      filters_per_class = tf.math.round(k/n_classes)        
      class_map = np.zeros((k),dtype=np.uint8)
      count=0
      for i in range(n_classes):
          for j in range(int(filters_per_class)):
              class_map[count]= i
              count+=1
              if count == k:
                  break    
      return class_map
#%% in loop individually
#t = -4 #target_conv_kernel # -4, -8 , -10, -12 ## conv layer indexes --> top to bottom
#b = -3 #target_conv_bias  #  -3, -7 , -9,  -11 ## bias layer indexes --> top to bottom
#filters=128  #               128  128   64   32
def save_filter_importance(model,weights_path,x,y_gt,label_map,original_prob,class_specific_prob,img_ind, args,save_path):

    class_specific = True
    print_output = True
    # if class_specific:
    #     print ("computing class-specific filter importance")
    # else:
    #     print ("computing arMax class filter importance")
    
    #for i in range(len(model.weights)):
    #    print(len(model.weights)-i,  model.weights[i].shape)
    
    if args.interpretable:
        layer_indexes = np.array(((-4, -3), (-8, -7),(-10, -9),(-12, -11) )) # first col=layer; 2nd col=bias
        filter_indexes = np.array((128, 128, 64, 32))
    else:
        if args.full_standard: ### for VGG mode #TODO: auotmate check/indexing 
            layer_indexes = np.array(((-4, -3),(-6, -5),(-8, -7),(-10, -9))) # first col=layer; 2nd col=bias
            filter_indexes = np.array((512, 512, 512))               
        else:
            layer_indexes = np.array(((-4, -3), (-6, -5),(-8, -7),(-10, -9) )) # first col=layer; 2nd col=bias
            filter_indexes = np.array((128, 128, 64, 32))        
    #order --> top to bottom
    #TODO: automate index creation
    #TODO: check that probability of setting a filter to zero in lew layer loop is correct ; compared with setting it outside the loop
            #there is discrepancy between manual evaluated probs and in-loop probs
    argMax_matrix = []
    argMax_pred_class = []
    argMax_stats = []

    class_specific_matrix = []
    class_specific_stats = []
    
        
    for k in range(len(layer_indexes)):
        t=layer_indexes[k][0]
        b=layer_indexes[k][1]
        filters = filter_indexes[k]
        
        list_probs = np.zeros((filters))#argMax class
        list_class = np.zeros((filters))#argMax class
        list_stats = np.zeros((7)) #argMax class
        
        c_list_probs = np.zeros((filters)) #class specific probs
        c_list_stats = np.zeros((8)) #class specific stats        
        
        for i in range(filters):
            model.load_weights(filepath=weights_path+'/model.hdf5')
            #pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
    
            original_weights = model.get_weights()
            modified_weights = np.copy(original_weights)
            
            modified_weights[t][:,:,:,i]=tf.zeros_like(modified_weights[t][:,:,:,i])
            modified_weights[b][i]=tf.zeros_like(modified_weights[b][i])
            model.set_weights(modified_weights)
    
            if args.full_standard:
                pred_probs = model(np.expand_dims(x,0),y_gt)#with eager
            else:                   
                pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x,0),y_gt)#with eager
            list_probs[i]=np.max(pred_probs) #argMax class
            list_class[i]=np.argmax(pred_probs) #argMax class

            c_list_probs[i]=pred_probs[0][np.argmax(y_gt)] #class specific
            if (class_specific and print_output):
                #print('layer_',k,'_filter_',i,' : gt class: ',label_map[np.argmax(y_gt)], '  prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%')
                string = 'layer_'+str(k)+'_filter_'+str(i)+' : gt class: '+label_map[np.argmax(y_gt)]+ '  prob: '+str(pred_probs[0][np.argmax(y_gt)].numpy()*100)+'%'
                sys.stdout.write("\r"+string)
                #sys.stdout.flush()

            #else:
            #    print('layer_',k,'_filter_',i,' : predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
            #print ('actual: ', label_map[np.argmax(y_gt)])
        ##%%
        if (class_specific and print_output):
            print('\nmin: ', np.min(c_list_probs))
            print('min_filter: ',np.argmin(c_list_probs))
        
            print('max: ',np.max(c_list_probs))
            print('max_filter: ',np.argmax(c_list_probs))
            
            print('std: ',np.std(c_list_probs))
            print('mean: ',np.mean(c_list_probs))
        # else:
        #     print('min: ', np.min(list_probs))
        #     print('min_filter: ',np.argmin(list_probs))
        
        #     print('max: ',np.max(list_probs))
        #     print('max_filter: ',np.argmax(list_probs))
            
        #     print('std: ',np.std(list_probs))
        #     print('mean: ',np.mean(list_probs))
        # print('\n')
        #argMax:
        list_stats[0]=np.min(list_probs)
        list_stats[1]=np.argmin(list_probs)
        list_stats[2]=np.max(list_probs)
        list_stats[3]=np.argmax(list_probs)
        list_stats[4]=np.std(list_probs)
        list_stats[5]=np.mean(list_probs)
        list_stats[6]=original_prob
  
        argMax_matrix.append(list_probs)
        argMax_pred_class.append(list_class)
        argMax_stats.append(list_stats)
        
        #class_specific:
        c_list_stats[0]=np.min(c_list_probs)
        c_list_stats[1]=np.argmin(c_list_probs)
        c_list_stats[2]=np.max(c_list_probs)
        c_list_stats[3]=np.argmax(c_list_probs)
        c_list_stats[4]=np.std(c_list_probs)
        c_list_stats[5]=np.mean(c_list_probs)
        c_list_stats[6]=class_specific_prob #original class_specific
        c_list_stats[7]=np.argmax(y_gt)

        class_specific_matrix.append(c_list_probs)
        class_specific_stats.append(c_list_stats)

        if args.save_top_layer:
            break        
        # end inner for loop
    
    argMax_delta_matrix=argMax_matrix-original_prob
    class_specific_delta_matrix=class_specific_matrix-class_specific_prob
    
    #matrix = np.asarray(matrix)
    #pred_class = np.asarray(pred_class)
    #stats = np.asarray(stats)
    #delta_matrix = np.asarray(delta_matrix)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    #argMax
    # np.save(file=save_path+'/'+str(img_ind)+'_matrix_argMax.npy',arr=argMax_matrix,allow_pickle=True)
    # np.save(file=save_path+'/'+str(img_ind)+'_pred_class_argMax.npy',arr=argMax_pred_class,allow_pickle=True)
    # np.save(file=save_path+'/'+str(img_ind)+'_stats_argMax.npy',arr=argMax_stats,allow_pickle=True)
    # np.save(file=save_path+'/'+str(img_ind)+'_delta_matrix_argMax.npy',arr=argMax_delta_matrix,allow_pickle=True)
    #class-specific
    np.save(file=save_path+'/'+str(img_ind)+'_matrix_class_specific.npy',arr=class_specific_matrix,allow_pickle=True)
    np.save(file=save_path+'/'+str(img_ind)+'_stats_class_specific.npy',arr=class_specific_stats,allow_pickle=True)
    np.save(file=save_path+'/'+str(img_ind)+'_delta_matrix_class_specific.npy',arr=class_specific_delta_matrix,allow_pickle=True)
       
    return argMax_matrix,argMax_pred_class,argMax_stats,argMax_delta_matrix, class_specific_matrix,class_specific_stats,class_specific_delta_matrix
    #outer loop end
#%%
def save_filter_importance_batch(model,weights_path,x,y_gt,label_map,original_prob,class_specific_prob, args):

    class_specific = True
    print_output = True
    # if class_specific:
    #     print ("computing class-specific filter importance")
    # else:
    #     print ("computing arMax class filter importance")
    
    #for i in range(len(model.weights)):
    #    print(len(model.weights)-i,  model.weights[i].shape)
    
    if args.interpretable:
        layer_indexes = np.array(((-4, -3), (-8, -7),(-10, -9),(-12, -11) )) # first col=layer; 2nd col=bias
        filter_indexes = np.array((128, 128, 64, 32))
    else:
        if args.full_standard: ### for VGG mode #TODO: auotmate check/indexing 
            layer_indexes = np.array(((-4, -3),(-6, -5),(-8, -7),(-10, -9))) # first col=layer; 2nd col=bias
            filter_indexes = np.array((512, 512, 512))               
        else:
            layer_indexes = np.array(((-4, -3), (-6, -5),(-8, -7),(-10, -9) )) # first col=layer; 2nd col=bias
            filter_indexes = np.array((128, 128, 64, 32))        
    #order --> top to bottom
    #TODO: automate index creation
    #TODO: check that probability of setting a filter to zero in lew layer loop is correct ; compared with setting it outside the loop
            #there is discrepancy between manual evaluated probs and in-loop probs

        
    for k in range(len(layer_indexes)):
        t=layer_indexes[k][0]
        b=layer_indexes[k][1]
        filters = filter_indexes[k]
        

        c_list_probs = np.zeros((len(x),filters)) #class specific probs
        c_list_stats = np.zeros((len(x),8)) #class specific stats        
        class_specific_delta_matrix = np.zeros((len(x),filters))
        
        for i in range(filters):
            model.load_weights(filepath=weights_path+'/model.hdf5')
            #pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
    
            original_weights = model.get_weights()
            modified_weights = np.copy(original_weights)
            
            modified_weights[t][:,:,:,i]=tf.zeros_like(modified_weights[t][:,:,:,i])
            modified_weights[b][i]=tf.zeros_like(modified_weights[b][i])
            model.set_weights(modified_weights)
    
            if args.full_standard:
                pred_probs, fmaps = model(x,y_gt)#with eager
            else:                   
                pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(x,y_gt)#with eager
            #list_probs[i]=np.max(pred_probs) #argMax class
            #list_class[i]=np.argmax(pred_probs) #argMax class

            for j in range(len(y_gt)):
                c_list_probs[j,i]=pred_probs[j,np.argmax(y_gt[j])] #class specific
            #if (class_specific and print_output):
            #    #print('layer_',k,'_filter_',i,' : gt class: ',label_map[np.argmax(y_gt)], '  prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%')
            #    string = 'layer_'+str(k)+'_filter_'+str(i)+' : gt class: '+label_map[np.argmax(y_gt)]+ '  prob: '+str(pred_probs[0][np.argmax(y_gt)].numpy()*100)+'%'
            sys.stdout.write("\r"+str(i))
            #    #sys.stdout.flush()
            # if i==3:
            #     break
            #else:
            #    print('layer_',k,'_filter_',i,' : predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
            #print ('actual: ', label_map[np.argmax(y_gt)])
        ##%%

        #class_specific:
        c_list_stats[:,0]=np.min(c_list_probs,axis=1)
        c_list_stats[:,1]=np.argmin(c_list_probs,axis=1)
        c_list_stats[:,2]=np.max(c_list_probs,axis=1)
        c_list_stats[:,3]=np.argmax(c_list_probs,axis=1)
        c_list_stats[:,4]=np.std(c_list_probs,axis=1)
        c_list_stats[:,5]=np.mean(c_list_probs,axis=1)
        for j in range(len(y_gt)):
            c_list_stats[j,6]=class_specific_prob[j,np.argmax(y_gt[j])] #original class_specific
        c_list_stats[:,7]=np.argmax(y_gt,axis=1)

        
        if args.save_top_layer:
            break        
        # end inner for loop
    for j in range(len(y_gt)):
        class_specific_delta_matrix[j,:]=c_list_probs[j,:] - class_specific_prob[j, np.argmax(y_gt[j])]
    
    #class_specific_delta_matrix=c_list_probs-np.repeat(class_specific_prob[:,np.argmax(y_gt)], (filters)).reshape(((len(x),filters)))
    
    #matrix = np.asarray(matrix)
    #pred_class = np.asarray(pred_class)
    #stats = np.asarray(stats)
    #delta_matrix = np.asarray(delta_matrix)
    

    return c_list_probs,c_list_stats,class_specific_delta_matrix
    #outer loop end
#%%
def plot_filter_importance(array, class_specific_stats):
    if class_specific_stats.ndim>1:
        print('original class probability:',class_specific_stats[0][6])
    else:
        print('original class probability:',class_specific_stats[6])
    
    if array.ndim>1:
        array = array[0]

        
    plt.plot(array), plt.title('delta_matrix - layer 0 (top)'),plt.show()
    #plt.plot(array[1]), plt.title('delta_matrix - layer 1'),plt.show()
    #plt.plot(array[2]), plt.title('delta_matrix - layer 2'),plt.show()
    #plt.plot(array[3]), plt.title('delta_matrix - layer 3'),plt.show()
    
#%%
def check_top_filter_importance(model,weights_path,x,y_gt,label_map,original_prob,class_specific_prob,img_ind,best_indexes,args):

   
        
    if args.interpretable:
        layer_indexes = np.array(((-4, -3), (-8, -7),(-10, -9),(-12, -11) )) # first col=layer; 2nd col=bias
    else:
        layer_indexes = np.array(((-4, -3), (-6, -5),(-8, -7),(-10, -9) )) # first col=layer; 2nd col=bias
        #order --> top to bottom
    #TODO: automate index creation

    model.load_weights(filepath=weights_path+'/model.hdf5')

    original_weights = model.get_weights()
    modified_weights = np.copy(original_weights)
    for k in range(len(layer_indexes)):
        t=layer_indexes[k][0]
        b=layer_indexes[k][1]
           
        
      
        
        i=best_indexes[k][0].astype(np.int32)
        modified_weights[t][:,:,:,i]=tf.zeros_like(modified_weights[t][:,:,:,i])
        modified_weights[b][i]=tf.zeros_like(modified_weights[b][i])
        model.set_weights(modified_weights)

    if args.full_standard:
        pred_probs,fmaps = model(np.expand_dims(x,0),y_gt)#with eager
    else:
        pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x,0),y_gt)#with eager
 
    print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%')
    #string = 'layer_'+str(k)+'_filter_'+str(i)+' : gt class: '+label_map[np.argmax(y_gt)]+ '  prob: '+str(pred_probs[0][np.argmax(y_gt)].numpy()*100)+'%'
        #sys.stdout.write("\r"+string)
#%%
def check_histogram_top_filter_result(model,filters,x,y_gt,label_map,args):

    if args.interpretable:
            layer_indexes = np.array(((-4, -3), (-8, -7),(-10, -9),(-12, -11) )) # first col=layer; 2nd col=bias
    else:
            layer_indexes = np.array(((-4, -3), (-6, -5),(-8, -7),(-10, -9) )) # first col=layer; 2nd col=bias
            #order --> top to bottom
        #TODO: automate index creation
       
    original_weights = model.get_weights()
    modified_weights = np.copy(original_weights)
    
    #for k in range(len(layer_indexes)):
    t=layer_indexes[0][0]
    b=layer_indexes[0][1]
       
    
  
    for j in range(len(filters)):
        modified_weights[t][:,:,:,filters[j]]=tf.zeros_like(modified_weights[t][:,:,:,j])
        modified_weights[b][filters[j]]=tf.zeros_like(modified_weights[b][j])
    model.set_weights(modified_weights)

    if args.full_standard:
        pred_probs,fmaps = model(np.expand_dims(x,0),y_gt)#with eager
    else:
        pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x,0),y_gt)#with eager
    
    print( 'gt class: ',label_map[np.argmax(y_gt)], '  prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%')
    #string = 'layer_'+str(k)+'_filter_'+str(i)+' : gt class: '+label_map[np.argmax(y_gt)]+ '  prob: '+str(pred_probs[0][np.argmax(y_gt)].numpy()*100)+'%'
        #sys.stdout.write("\r"+string)  
    return model, fmaps
#%%
def test_filter_importance(model,weights_path,x,y_gt,label_map,original_prob,img_ind):
    for i in range(len(model.weights)):
        print(len(model.weights)-i,  model.weights[i].shape)
    #%% in loop individually

    t = -4 #target_conv_kernel # -4, -8 , -10, -12 ## conv layer indexes
    b = -3 #target_conv_bias  #  -3, -7 , -9,  -11 ## bias layer indexes
    filters=512  #               128  128   64   32
    list_probs = np.zeros((filters))
    for i in range(filters):
        model.load_weights(filepath=weights_path+'/model.hdf5')
        original_weights = model.get_weights()
        modified_weights = np.copy(original_weights)
        
        modified_weights[t][:,:,:,i]=tf.zeros_like(modified_weights[t][:,:,:,i])
        modified_weights[b][i]=tf.zeros_like(modified_weights[b][i])
        model.set_weights(modified_weights)

        pred_probs = model(np.expand_dims(x,0),y_gt)#with eager
        #pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x,0),y_gt)#with eager
        list_probs[i]=np.max(pred_probs)
        print('filter_',i,' : predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
        #print ('actual: ', label_map[np.argmax(y_gt)])
    ##%%
    print('min: ', np.min(list_probs))
    print('min_filter: ',np.argmin(list_probs))

    print('max: ',np.max(list_probs))
    print('max_filter: ',np.argmax(list_probs))
    
    print('std: ',np.std(list_probs))
    print('mean: ',np.mean(list_probs))
    #%% in loop filter class-wise
    class_map = get_class_map(filters=128,num_classes=10)
    t= -4
    b = -3
    filters=128
    target_off = True #default: True --> target off and rest on; or reverse

    for i in range(1):
        model.load_weights(filepath=weights_path+'/model.hdf5')
        original_weights = model.get_weights()
        modified_weights = np.copy(original_weights)
         
        class_ind = tf.where(class_map==i)
        if target_off:
            modified_weights[t][:,:,:,class_ind[0][0]:class_ind[-1][0]+1]=tf.zeros_like(modified_weights[t][:,:,:,class_ind[0][0]:class_ind[-1][0]+1])
            modified_weights[b][class_ind[0][0]:class_ind[-1][0]+1]=tf.zeros_like(modified_weights[b][class_ind[0][0]:class_ind[-1][0]+1])
        else:
            modified_weights[t] = np.zeros_like(original_weights[t])
            modified_weights[b] = np.zeros_like(original_weights[b])
 
            modified_weights[t][:,:,:,class_ind[0][0]:class_ind[-1][0]+1]=original_weights[t][:,:,:,class_ind[0][0]:class_ind[-1][0]+1]
            modified_weights[b][class_ind[0][0]:class_ind[-1][0]+1]=original_weights[b][class_ind[0][0]:class_ind[-1][0]+1]

            
        model.set_weights(modified_weights)
        
        pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x,0),y_gt)#with eager
    
        print('class_',i,' : predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
        #print ('actual: ', label_map[np.argmax(y_gt)])  
    #%% for fixed classes

    fig, axs = plt.subplots(32,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
    #fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    for i in range(128):
    
        #axs[i].imshow(fmaps_x1[0,:,:,i],cmap='gray',vmin=0, vmax=1)#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
        axs[i].imshow(fmaps_x2[0,:,:,i].numpy().squeeze(),cmap='gray')
        axs[i].axis('off')
    
    #plt.title('templates_1')    
    plt.show()
#%% in-code method to modify filters 
def test_filter_importance_in_code_method(model,weights_path,x,y_gt,label_map,img_ind):
    ##%% in loop individually
    class_map = get_class_map(filters=128,num_classes=10)
    t = -4 #target_conv_kernel # -4, -8 , -10, -12 ## conv layer indexes
    b = -3 #target_conv_bias  #  -3, -7 , -9,  -11 ## bias layer indexes
    filters=128  #               128  128   64   32
    list_probs = np.zeros((filters))
            
    for i in range(filters):
        filters_disabled = i
        
        pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x,0),y_gt,filters_disabled=filters_disabled)#with eager
        list_probs[i]=np.max(pred_probs)

        print('class_',i,' : predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
        #print ('actual: ', label_map[np.argmax(y_gt)])  
    ##%%
    print('min: ', np.min(list_probs))
    print('min_filter: ',np.argmin(list_probs))

    print('max: ',np.max(list_probs))
    print('max_filter: ',np.argmax(list_probs))
    
    print('std: ',np.std(list_probs))
    print('mean: ',np.mean(list_probs))
    #%% in loop filter class-wise
    class_map = get_class_map(filters=128,num_classes=10)
    t = -10 #target_conv_kernel # -4, -8 , -10, -12 ## conv layer indexes
    b = -9 #target_conv_bias  #  -3, -7 , -9,  -11 ## bias layer indexes
    filters=64  #               128  128   64   32
    list_probs = np.zeros((filters))    
    for i in range(10):
        class_ind = tf.where(class_map==i)
        filters_disabled = class_ind
        #all_disabled = filters_disabled.numpy()
        #all_disabled[0]=0
        #all_disabled[-1]=127
        
        

        pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x,0),y_gt,filters_disabled=filters_disabled)#with eager
    
        print('class_',i,' : predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
        #print ('actual: ', label_map[np.argmax(y_gt)])  
        

    
 