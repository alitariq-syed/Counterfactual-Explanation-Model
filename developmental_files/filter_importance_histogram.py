# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:22:44 2020

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
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Softmax, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import math
import argparse
import os
from sklearn.model_selection import train_test_split

from tensorflow.keras import optimizers

from models10 import MySubClassModel

from codes.compute_filter_importance import save_filter_importance, test_filter_importance,test_filter_importance_in_code_method, plot_filter_importance,check_top_filter_importance,check_histogram_top_filter_result
from codes.load_cxr_dataset import create_cxr_dataframes, load_cxr_dataset

#%%
parser = argparse.ArgumentParser(description='Filter_importance_CNN')
parser.add_argument('--interpretable',default = False)
parser.add_argument('--resume', default =True) # load saved weights
parser.add_argument('--pretrained', default = False)
parser.add_argument('--filter_modified_directly', default = True)
parser.add_argument('--loss_compute', default = True)#False = forward only
parser.add_argument('--high_capacity_model', default = True)#
parser.add_argument('--fixed_classes', default = True)#idea 2: fine tune from forward only with fixed classes
parser.add_argument('--fixed_classes_reduce_loss', default = True)#False = forward only masked with fixed filter class. issue: 100% training accuracy but 10% testing acc

parser.add_argument('--dataset',default = 'cifar10')#mnist, cifar10, CUB200, #cxr1000
parser.add_argument('--save_directory',default = './trained_weights/')
parser.add_argument('--train',default = False)
parser.add_argument('--test', default = False)
parser.add_argument('--model',default = 'myCNN/')#myCNN, VGG16
parser.add_argument('--imagenet_weights',default = False)
parser.add_argument('--filter_category_method',default = 'own_reduce_loss')   # paper --> similar to paper implementation---assign filter with categories during training by accumulating batch-wise max activations
                                                                    # own_reduce_loss --> our idea - pre-assign filter categories during forward pass over all the data, based on pretrained weights and feature maps

#parser.add_argument('--test',default = True)
 
args = parser.parse_args()

if args.interpretable:
    if args.filter_category_method=='paper':
        print('filter category assignment --> paper method')
    else:
        print('filter category assignment --> our idea')
    weights_path = args.save_directory+args.model+args.dataset+'/interpretable/filter_category_method_'+str(args.filter_category_method)
    log_path  = './logs/'+args.model+args.dataset+'/interpretable/filter_category_method_'+str(args.filter_category_method)
else:
    weights_path = args.save_directory+args.model+args.dataset+'/standard'
    log_path  = './logs/'+args.model+args.dataset+'/standard'
    
if not os.path.exists(weights_path):
    os.makedirs(weights_path)    
print('save_path: ',weights_path)

parser.add_argument('--save_path',default = weights_path)
args = parser.parse_args()

if args.resume:
    print("resuming training")
#print(args)
#%%
batch_size = 32
if args.dataset == 'mnist':
    num_classes=10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train,-1)
    x_test = np.expand_dims(x_test,-1)
    input_shape = (batch_size,28,28,1)
    label_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

elif args.dataset == 'cifar10':
    num_classes=10
    print('cifar-10 dataset')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_shape = (batch_size,32,32,3)
    label_map = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

elif args.dataset == 'CUB200':
    print('CUB200-2011 dataset')
    num_classes=200
    #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_shape = (batch_size,224,224,3)
    #label_map = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    data_dir ='G:/CUB_200_2011/CUB_200_2011/images/'
    label_map = np.loadtxt('G:/CUB_200_2011/CUB_200_2011/classes.txt',dtype='str')
    label_map = label_map[:,1]

elif args.dataset == 'cxr1000':
    print('CXR-1000 dataset')
    num_classes=15
    input_shape = (batch_size,224,224,3)
    label_map, train_df, test_df, valid_df = create_cxr_dataframes()
    all_labels = label_map

#%%
if args.dataset == 'cxr1000':
    train_gen, test_gen, valid_gen = load_cxr_dataset(train_df, test_df, valid_df, all_labels, batch_size)
    
else:

    imgDataGen = ImageDataGenerator(rescale = 1./255)

    train_gen = imgDataGen.flow(x_train, y_train, batch_size = batch_size,shuffle= False)#Normally it hould be true.... but for filter importance work it is set to false so that saved files can be correctly matched with respective images
    test_gen  = imgDataGen.flow(x_test, y_test, batch_size = batch_size,shuffle= False)


#%% load filter importance matrices for training

class_for_analysis = 0
print ('class for analysis: ', label_map[class_for_analysis])
trainSplit = True# train or test split

save_path = './create_training_data/'+ str(class_for_analysis)
if trainSplit:
    save_path+=' - train_split'
    batches=math.ceil(train_gen.n/train_gen.batch_size)

    train_gen.reset() #resets batch index to 0
    gen=train_gen
else:
    save_path+=' - test_split'
    batches=math.ceil(test_gen.n/test_gen.batch_size)

    test_gen.reset() #resets batch index to 0
    gen=test_gen
hist_positive=[]
hist_negative=[]

scaled = False
spikes = False

gt_bin_positive=np.zeros((128))
gt_bin_negative=np.zeros((128))

orig_bin_positive=np.zeros((128))
orig_bin_negative=np.zeros((128))

for k in range(batches):
    x_batch_test,y_batch_test = next(gen)

    for i in range (len(x_batch_test)):
        img_ind = i#3
        actual_img_ind = i + (gen.batch_index-1)*gen.batch_size

        y_gt = y_batch_test[img_ind]
        

        #load saved arrays        
        if os.path.exists(path=save_path+'/'+str(actual_img_ind)+'_delta_matrix_class_specific.npy'):
            print('img_ind:',actual_img_ind)
        else:
            continue
        
        # plt.figure(figsize = (20,2))
        # plt.imshow(x_batch_test[img_ind].squeeze(),cmap='gray')
        # plt.axis('off')
        # plt.show()
                 
        #model.load_weights(filepath=weights_path+'/model.hdf5')    
        #pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
        #print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
        #print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%\n\n')
        
        #class-specific
        class_specific_matrix = np.load(file=save_path+'/'+str(actual_img_ind)+'_matrix_class_specific.npy',allow_pickle=True)
        class_specific_stats = np.load(file=save_path+'/'+str(actual_img_ind)+'_stats_class_specific.npy',allow_pickle=True)
        class_specific_delta_matrix = np.load(file=save_path+'/'+str(actual_img_ind)+'_delta_matrix_class_specific.npy',allow_pickle=True)

        #print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
        
        #original_prob = np.max(pred_probs)
        #class_specific_prob = pred_probs[0][np.argmax(y_gt)].numpy()
    
        #print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',class_specific_prob*100,'%')
        if np.sum(class_specific_delta_matrix[0])==0:
            print('zero variance - img_ind:',actual_img_ind)
        else:#skip it for training
            img = x_batch_test[img_ind]
            gt = class_specific_delta_matrix[0]
            
           # plt.plot(gt), plt.title('delta_matrix - gt'),plt.show()
            
            #only get spikes
            if spikes:
                #gt[gt<np.std(gt)]=0
                gt_min = gt.copy()
                gt_max = gt.copy()
    
                #if np.min(gt)<0 and np.max(gt)>0:
                gt_min[gt>-np.std(gt)]=0
                gt_max[gt<np.std(gt)] =0
                
                gt_spikes = gt_min+gt_max
                gt = gt_spikes

            #spike -1,0,1 only
            #gt[gt<0]=-1
            #gt[gt>0]=1

               
            #scaling -1 to 1
            if scaled:
                gt = gt-np.min(gt)# get min to zero
                gt = gt/np.max(gt)*2 # get max to 2
                gt = gt - 1 # get -1 to 1
            

            #plt.plot(gt_min), plt.title('delta_matrix - spikes'),plt.show()
            #plt.plot(gt_max), plt.title('delta_matrix - spikes'),plt.show()
            #plt.plot(gt), plt.title('delta_matrix - spikes'),plt.show()
            
            #get data for histogram
            hist_thresh = 0.1
            
            gt_bin_negative[gt<=-hist_thresh]+=1
            gt_bin_positive[gt>=hist_thresh] +=1
            
            hist_positive.append(gt_bin_positive)
            hist_negative.append(gt_bin_negative)
            
            orig_bin_negative[gt<=-hist_thresh]+= class_specific_stats[0][6]*100
            orig_bin_positive[gt>=hist_thresh] += class_specific_stats[0][6]*100

        
        #end inner loop
    #end outer loop
#%%
count_threshold=125
plt.plot(gt_bin_negative),plt.title('gt_bin_negative - ' +str(hist_thresh)),
plt.hlines(count_threshold,xmin=0,xmax=128,colors='r'), plt.show()

plt.plot(gt_bin_positive),plt.title('gt_bin_positive - ' +str(hist_thresh)),
plt.hlines(80,xmin=0,xmax=128,colors='r'), plt.show()

avg_orig_bin_negative = np.nan_to_num(orig_bin_negative/gt_bin_negative)
avg_orig_bin_positive = np.nan_to_num(orig_bin_positive/gt_bin_positive)

plt.plot(avg_orig_bin_negative),plt.title('avg_orig_bin_negative - ' +str(hist_thresh)),plt.show()
plt.plot(avg_orig_bin_positive),plt.title('avg_orig_bin_positive - ' +str(hist_thresh)),plt.show()

hist_positive = np.asarray(hist_positive)
hist_negative = np.asarray(hist_negative)

#%%

neg_filters = np.where(gt_bin_negative>=count_threshold)
pos_filters = np.where(gt_bin_positive>=40)

#[ 4, 23, 69, 74] #pos
# [ 24,  34,  99, 113, 122] #neg

#%%
if args.high_capacity_model:
    def MyFunctionalModel():
        inputs = tf.keras.Input(shape = train_gen.x[0].shape)
      
        x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPool2D((2,2))(x)
        x = Conv2D(128, (3, 3), activation='relu',strides=(1,1), name='target_conv')(x)
    
        return tf.keras.Model(inputs, x)
else:
    def MyFunctionalModel():
        inputs = tf.keras.Input(shape = train_gen.x[0].shape)
      
        x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPool2D((2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu',strides=(1,1), name='target_conv')(x)
    
        return tf.keras.Model(inputs, x)

#%%
if args.dataset == 'cxr1000':
    print('loading VGG model')

    tr = 1
    if tr:
        print('using imagenet weights')
        vgg = VGG16(weights='imagenet',include_top = True)#top needed to get output dimensions at each layer
            # EfficientNetB0(include_top=True,
            #                weights=None,
            #                input_shape=img_shape,
            #                classes=len(all_labels),
            #                classifier_activation='sigmoid')
        base_model = tf.keras.Model(vgg.input,vgg.layers[-6].output)
        #model.compile(optimizer = optimizers.RMSprop(), loss = 'binary_crossentropy',#adam #weighted_binary_crossentropy #lr=0.001/2#binary_crossentropy
                                   #metrics = ['binary_accuracy'])#,tf.keras.metrics.AUC()])
    else:
        print('loading saved model - NOT IMPLEMENTED YET')
        #model = load_model('../input/efnb0-saved-weights/xray_class_EfficientNetB4_15_class_CEL_heatmap_imagenet_pretrained_weights.05-0.1807.hdf5')
    
    #model.summary()
else:
    #base_model = VGG16(weights=None,include_top = False)
    base_model = MyFunctionalModel()

#%%
model = MySubClassModel(num_classes=num_classes, base_model=base_model, args=args)
#model = base_model
model(tf.zeros(input_shape))
#model.build(input_shape = input_shape)

model.summary()

#load saved weights
if args.resume:
    #model.load_weights('./trained_weights/myCNN/cifar10/standard/model.hdf5')
    #model.load_weights('./trained_weights/myCNN/cifar10/interpretable/filter_category_method_paper/from_pretrained_model.hdf5')
    model.load_weights(filepath=weights_path+'/model.hdf5')
    #model.load_weights('./trained_weights/myCNN/cifar10/interpretable/filter_category_method_paper/model.hdf5')

    print("weights loaded")
if args.pretrained:
    model.load_weights('./trained_weights/myCNN/cifar10/standard/model.hdf5')
    print("pretrained weights loaded")



#%% analyze_filter_importance
class_for_analysis = 0
print ('class for analysis: ', label_map[class_for_analysis])

save_path = './create_training_data/'+ str(class_for_analysis)

batches=math.ceil(test_gen.n/test_gen.batch_size)

print('top neg filters', neg_filters)
print('top pos filters', pos_filters)

test_gen.reset() #resets batch index to 0
for k in range(batches):
    x_batch_test,y_batch_test = next(test_gen)

    for i in range (len(x_batch_test)):
        img_ind = i#3
        actual_img_ind = i + (test_gen.batch_index-1)*test_gen.batch_size

        y_gt = y_batch_test[img_ind]
        
         #skip other class        
        if class_for_analysis==np.argmax(y_gt):
            print('\n\nimg_ind:',actual_img_ind)
        else:
            continue
        
        # plt.figure(figsize = (20,2))
        # plt.imshow(x_batch_test[img_ind].squeeze(),cmap='gray')
        # plt.axis('off')
        # plt.show()
                 
        model.load_weights(filepath=weights_path+'/model.hdf5')    
        pred_probs,fmaps_x1,fmaps_x2,target_1,target_2,raw_map,forward_1 = model(np.expand_dims(x_batch_test[img_ind],0),y_batch_test[img_ind])#with eager
        print('predicted: ',label_map[np.argmax(pred_probs)], ' with prob: ',np.max(pred_probs)*100,'%')
        print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',pred_probs[0][np.argmax(y_gt)].numpy()*100,'%')
        
        print('neg')
        check_histogram_top_filter_result(model,neg_filters,x_batch_test[img_ind],y_gt,label_map,args)
        
        model.load_weights(filepath=weights_path+'/model.hdf5')    
        print('pos')
        check_histogram_top_filter_result(model,pos_filters,x_batch_test[img_ind],y_gt,label_map,args)
        y_gt
       # break
    #break