# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:36:24 2020

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
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Softmax, Dropout, GlobalAveragePooling2D,BatchNormalization
#from tensorflow.keras.reg
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import math
import argparse
import os, sys
from sklearn.model_selection import train_test_split

from tensorflow.keras import optimizers

from codes.compute_filter_importance import save_filter_importance, test_filter_importance,test_filter_importance_in_code_method, plot_filter_importance,check_top_filter_importance
from codes.load_cxr_dataset import create_cxr_dataframes, load_cxr_dataset

from tensorflow.keras.applications.vgg16 import VGG16,decode_predictions, preprocess_input


#%%
parser = argparse.ArgumentParser(description='Filter_importance_CNN')
parser.add_argument('--resume', default =False) # load saved weights

parser.add_argument('--train_filter_importance', default = True)#for testing idea 2
parser.add_argument('--train_filter_fmaps', default = False)#for testing idea 2


parser.add_argument('--dataset',default = 'catsvsdogs')#mnist, cifar10, CUB200, #cxr1000, #catsvsdogs
parser.add_argument('--save_directory',default = './trained_weights/')
parser.add_argument('--train',default = True)
parser.add_argument('--test', default = False)
parser.add_argument('--model',default = 'VGG16/')#myCNN, VGG16


#parser.add_argument('--test',default = True)
 
args = parser.parse_args()

weights_path = args.save_directory+args.model+args.dataset+'/filter_importance_CNN'
log_path  = './logs/'+args.model+args.dataset+'/filter_importance_CNN'
    
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
elif args.dataset == 'catsvsdogs':
    print('catsvsdogs dataset')
    num_classes=2
    input_shape = (batch_size,224,224,3)
    data_dir ='G:/catsvsdogs/train/'
    data_dir_test ='G:/catsvsdogs/test/'

    label_map = ['cat',  'dog']

else:
    print('unknown dataset')
    sys.exit()
#%%
if args.dataset == 'cxr1000':
    train_gen, test_gen, valid_gen = load_cxr_dataset(train_df, test_df, valid_df, all_labels, batch_size)
elif args.dataset == 'catsvsdogs':
    imgDataGen = ImageDataGenerator(preprocessing_function = preprocess_input, 
                                #rescale = 1./255,
                                validation_split=0.2)
    
    train_gen = imgDataGen.flow_from_directory(data_dir,
                        target_size=(input_shape[1], input_shape[2]),
                        color_mode='rgb',
                        class_mode='categorical',
                        batch_size=batch_size,
                        shuffle=False,
                        seed=None,
                        subset='training',
                        interpolation='nearest')
    test_gen  = imgDataGen.flow_from_directory(data_dir,
                        target_size=(input_shape[1], input_shape[2]),
                        color_mode='rgb',
                        class_mode='categorical',
                        batch_size=batch_size,
                        shuffle=False,
                        seed=None,
                        subset='validation',
                        interpolation='nearest')
else:

    imgDataGen = ImageDataGenerator(rescale = 1./255)

    train_gen = imgDataGen.flow(x_train, y_train, batch_size = batch_size,shuffle= False)#Normally it hould be true.... but for filter importance work it is set to false so that saved files can be correctly matched with respective images
    test_gen  = imgDataGen.flow(x_test, y_test, batch_size = batch_size,shuffle= False)

#%% load filter importance matrices for training
if args.train_filter_importance:
    class_for_analysis = 0
    print ('class for analysis: ', label_map[class_for_analysis])
    trainSplit = False# train or test split
    
    #save_path = './create_training_data/'+ str(class_for_analysis)
    save_path = './create_training_data/'+args.model+args.dataset+'/standard/'+ str(class_for_analysis)
    
    save_path_fmap = './create_training_data/fmaps/'+ str(class_for_analysis)

    if trainSplit:
        save_path+=' - train_split'
        save_path_fmap+=' - train_split'
        batches=math.ceil(train_gen.n/train_gen.batch_size)
    
        train_gen.reset() #resets batch index to 0
        gen=train_gen
    else:
        save_path+=' - test_split'
        save_path_fmap+=' - test_split'
        batches=math.ceil(test_gen.n/test_gen.batch_size)
    
        test_gen.reset() #resets batch index to 0
        gen=test_gen
    x_train_filterCNN=[]
    y_train_filterCNN=[]
    
    hist_positive=[]
    hist_negative=[]

    gt_bin_positive=np.zeros((512))
    gt_bin_negative=np.zeros((512))

    scaled = False
    spikes = False
    
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
             #load saved arrays        
            if args.train_filter_fmaps:
                if os.path.exists(path=save_path_fmap+'/'+str(actual_img_ind)+'_fmap_class_specific.npy'):
                    print('fmap doesnt exist:',actual_img_ind)
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
            if class_specific_delta_matrix.ndim>1:
                 gt = class_specific_delta_matrix[0]
            else:
                 gt = class_specific_delta_matrix
            #print ('actual: ', label_map[np.argmax(y_gt)], ' with prob: ',class_specific_prob*100,'%')
            if np.sum(gt)==0:
                print('zero variance - img_ind:',actual_img_ind)
            else:#skip it for training
                img = x_batch_test[img_ind]
 
                
                #plt.plot(gt), plt.title('delta_matrix - gt'),plt.show()
                
                #only get spikes
                if spikes:
                    #gt[gt<np.std(gt)]=0
                    gt_min = gt.copy()
                    gt_max = gt.copy()
        
                    # #std based threshold
                    gt_min[gt>-np.std(gt)]=0
                    gt_max[gt<np.std(gt)] =0
                    
                    #fixed threshold
                    # t=0.5
                    # gt_min[gt>-t]=0
                    # gt_max[gt<t] =0
                    
                    gt_spikes = gt_min+gt_max
                    gt = gt_spikes
                
                    #spike -1,0,1 only
                    gt[gt<0]=1#-1
                    gt[gt>0]=1#1
                
                #if no spike in the threshold range then skip
                if np.mean(gt)==0.0:
                    print('zero mean')
                    continue
                   
                #scaling -1 to 1
                if scaled:
                    gt = gt-np.min(gt)# get min to zero
                    gt = gt/np.max(gt)*2 # get max to 2
                    gt = gt - 1 # get -1 to 1
                
    
                #plt.plot(gt_min), plt.title('delta_matrix - spikes'),plt.show()
                #plt.plot(gt_max), plt.title('delta_matrix - spikes'),plt.show()
                #plt.plot(gt), plt.title('delta_matrix - spikes'),plt.show()
                
                
                #get data for histogram
                hist_thresh = 0.001
                
                gt_bin_negative[gt<=-hist_thresh]+=1
                gt_bin_positive[gt>=hist_thresh] +=1
                
                
                if args.train_filter_fmaps:
                    fmap = np.load(file=save_path_fmap+'/'+str(actual_img_ind)+'_fmap_class_specific.npy',allow_pickle=True)
                    x_train_filterCNN.append(fmap.squeeze())
                else:
                    x_train_filterCNN.append(img)
                y_train_filterCNN.append(gt)
                
                plot_fmap=False
                if (args.train_filter_fmaps and plot_fmap):
                    fig, axs = plt.subplots(8,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
                    #fig.subplots_adjust(hspace = .5, wspace=.001)
                    
                    axs = axs.ravel()
                    
                    for i in range(32):
                    
                        axs[i].imshow(fmap[0,:,:,i],cmap='gray')#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
                        axs[i].axis('off')
                        
                    plt.show()
                
            #
            
            #end inner loop
        #end outer loop
    x_train_filterCNN = np.asarray(x_train_filterCNN)
    y_train_filterCNN = np.asarray(y_train_filterCNN)

    if args.train_filter_fmaps:#normalize
        x_train_filterCNN/=np.max(x_train_filterCNN)
#%%
count_threshold=250
plt.plot(gt_bin_negative),plt.title('gt_bin_negative - ' +str(hist_thresh)),
#plt.hlines(count_threshold,xmin=0,xmax=512,colors='r'), 
plt.show()

plt.plot(gt_bin_positive),plt.title('gt_bin_positive - ' +str(hist_thresh)),
#plt.hlines(150,xmin=0,xmax=512,colors='r'), 
plt.show()

neg_filters = np.where(gt_bin_negative>=count_threshold)
pos_filters = np.where(gt_bin_positive>=40)

#%% train test split

x_train_filterCNN,x_test_filterCNN,y_train_filterCNN,y_test_filterCNN = train_test_split(x_train_filterCNN,y_train_filterCNN,test_size=0.10, random_state=42)
#%%
trainDataGen = ImageDataGenerator(rotation_range=15, width_shift_range=0.15,
    height_shift_range=0.15, shear_range=0.15, zoom_range=0.15,
  fill_mode='nearest', cval=0.0, horizontal_flip=False,
    vertical_flip=False, rescale=None, preprocessing_function=None,
    data_format=None, validation_split=0.0)
imgDataGen = ImageDataGenerator()#, images already scaled 0-1

train_gen_filterCNN = imgDataGen.flow(x_train_filterCNN, y_train_filterCNN, batch_size = batch_size,shuffle= True)
test_gen_filterCNN = imgDataGen.flow(x_test_filterCNN, y_test_filterCNN, batch_size = batch_size,shuffle= True)

#%%
visualize_cxr_images=0
if visualize_cxr_images:
    t_x, t_y = next(train_gen)
    fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
    for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
        c_ax.imshow(c_x[:,:,0], cmap = 'bone')#, vmin = -1.5, vmax = 1.5)
        c_ax.set_title(', '.join([n_class for n_class, n_score in zip(all_labels, c_y) 
                                 if n_score>0.5]))
        c_ax.axis('off')

#%%
vgg_model=True
if args.train_filter_fmaps:
    def MyCNNModel():
         inputs = tf.keras.Input(shape = train_gen_filterCNN.x[0].shape)
       
         #x = Conv2D(256, kernel_size=(3, 3), activation='relu')(inputs)
         #x = Conv2D(512, (3, 3), activation='relu')(x)
         #x = Dropout(rate=0.8)(x)
         #x = MaxPool2D((2,2))(x)
         ##x = Conv2D(128, (3, 3), activation='relu',strides=(1,1), name='target_conv')(x)
         ##x = MaxPool2D((2,2))(x)
         #x = Flatten()(x)
         x = GlobalAveragePooling2D()(inputs)
    
         x = Dropout(rate=0.8)(x)
    
         #x = Dense(512, activation = 'relu')(x)
         #x = Dense(256, activation = 'relu')(x)

         x = Dense(len(gt), activation = 'sigmoid')(x)#tanh
    
         return tf.keras.Model(inputs, x)
elif vgg_model:
    def MyCNNModel():
        print('using imagenet weights for catsvsdogs dataset')
        vgg = VGG16(weights='imagenet',include_top = False,input_shape=(224,224,3))#top needed to get output dimensions at each layer
        freeze=True
        if freeze:
            for layer in vgg.layers:
                layer.trainable = False
        base_model = tf.keras.Model(vgg.input,vgg.layers[-2].output)
        
        x =  MaxPool2D()(base_model.output)
        x = GlobalAveragePooling2D()(x)
        x = Dense(len(gt), activation = 'sigmoid',kernel_regularizer='l2')(x)#tanh
    
        return tf.keras.Model(base_model.input, x)
else:
    def MyCNNModel():
        inputs = tf.keras.Input(shape = train_gen_filterCNN.x[0].shape)
      
        x = Conv2D(32, kernel_size=(3, 3), activation='relu',kernel_regularizer='l2')(inputs)
        x=BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu',kernel_regularizer='l2')(x)
        
        x = MaxPool2D((2,2))(x)
        x=BatchNormalization()(x)

        x = Conv2D(128, (3, 3), activation='relu',strides=(1,1), name='target_conv',kernel_regularizer='l2')(x)
        x = MaxPool2D((2,2))(x)
        x = Flatten()(x)
    
        x = Dropout(rate=0.8)(x)
    
        #x = Dense(256, activation = 'relu')(x)
        
        x = Dense(len(gt), activation = 'sigmoid',kernel_regularizer='l2')(x)#tanh
    
        return tf.keras.Model(inputs, x)

#%%
model = MyCNNModel()
model.compile(optimizer = optimizers.RMSprop(learning_rate=0.001/2), loss = 'MAE',
                                   metrics = ['accuracy'])#MAE #binary_crossentropy
model.summary()

#%%
#load saved weights
if args.resume:
    #model.load_weights('./trained_weights/myCNN/cifar10/standard/model.hdf5')
    #model.load_weights('./trained_weights/myCNN/cifar10/interpretable/filter_category_method_paper/from_pretrained_model.hdf5')
    model.load_weights(filepath=weights_path+'/model.hdf5')
    #model.load_weights('./trained_weights/myCNN/cifar10/interpretable/filter_category_method_paper/model.hdf5')

    print("weights loaded")


#%% Trains for 5 epochs.

history = model.fit(train_gen_filterCNN, epochs=50, verbose=1, callbacks=None, validation_data=test_gen_filterCNN, shuffle=False)
model.save_weights(filepath=weights_path+'/model.hdf5')

#%%
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#%% check predicted matrices (from training data)
train_gen_filterCNN.reset()
x,y = next(train_gen_filterCNN)
print(np.min(x))
print(np.max(y))

y_pred = model.predict(x)

for i in range(20):#len(x)
    plt.plot(y[i]), plt.title('delta_matrix - layer 0 - GT'),plt.show()
    plt.plot(y_pred[i]), plt.title('delta_matrix - predicted'),plt.show()

#%% check predicted matrices (from testing data)
test_gen_filterCNN.reset()
x,y = next(test_gen_filterCNN)
print(np.min(x))
print(np.max(y))

y_pred = model.predict(x)

for i in range(20):#len(x)
    plt.plot(y_train_filterCNN[i]), plt.title('delta_matrix - layer 0 - GT'),plt.show()
    #plt.plot(y_pred[i]), plt.title('delta_matrix - predicted'),plt.show()
#%%
# test_scores = model.evaluate(test_gen, verbose=1)
# print('Test loss:', test_scores[0])
# print('Test accuracy:', test_scores[1])

