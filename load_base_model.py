# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 23:50:34 2022

@author: Ali
"""
#%%
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Softmax, GlobalAveragePooling2D

from config import args
#%%

if args.model == 'VGG16/' or args.model =='myCNN/':
    from tensorflow.keras.applications.vgg16 import VGG16,decode_predictions, preprocess_input
elif args.model == 'resnet50/':
    from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
elif args.model == 'efficientnet/':
    from tensorflow.keras.applications.efficientnet import EfficientNetB0, decode_predictions, preprocess_input
elif args.model == 'inceptionv3/':
    from tensorflow.keras.applications.inception_v3 import InceptionV3, decode_predictions, preprocess_input
else:
    print("incorrect model specified")
    sys.exit()    

#%%
if args.imagenet_weights:
    print('loading VGG model')
    if args.dataset == 'cxr1000' and False:
        tr = 1
        if tr:
            print('using imagenet weights for CXR dataset')
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
    elif args.dataset == 'cifar10':
        print('using imagenet weights for cifar10 dataset')
        vgg = VGG16(weights='imagenet',include_top = False,input_shape=(32,32,3))#top needed to get output dimensions at each layer
        freeze=True
        if freeze:
            for layer in vgg.layers:
                layer.trainable = False
        base_model = tf.keras.Model(vgg.input,vgg.layers[-2].output)
    elif args.dataset == 'CUB200' or args.dataset == 'cxr1000' or args.dataset == 'NIST':
        if args.model == 'VGG16/':
           print('using VGG16 imagenet weights for CUB200 dataset')
           vgg = VGG16(weights='imagenet',include_top = False,input_shape=(224,224,3))#top needed to get output dimensions at each layer
           freeze=True
           if freeze:
               for layer in vgg.layers:
                   # print (layer.name)
                   if layer.name == '----block5_conv3': continue
                   else: layer.trainable = False
                   
               neuron_sum=0    
               for layer in vgg.layers:
                    # print (layer.name, layer.output.shape)
                    if 'conv' in layer.name: 
                        # print (layer.name,'\t\t', layer.output.shape)
                        neuron_sum+=layer.output.shape[-1]
               print("total neurons: ", neuron_sum)                   
           base_model = tf.keras.Model(vgg.input,vgg.layers[-2].output)
        elif args.model == 'resnet50/':
           print('using resnet50 imagenet weights for CUB200 dataset')
           vgg = ResNet50(weights='imagenet',include_top = False,input_shape=(224,224,3))#top needed to get output dimensions at each layer
           freeze=True
           if freeze:
               for layer in vgg.layers:
                   print (layer.name)
                   if layer.name == '----block5_conv3': continue
                   else: layer.trainable = False
           base_model = tf.keras.Model(vgg.input,vgg.output)
        elif args.model == 'efficientnet/':
               print('using efficientnet imagenet weights for CUB200 dataset')
               vgg = EfficientNetB0(weights='imagenet',include_top = False,input_shape=(224,224,3))#top needed to get output dimensions at each layer
               freeze=True
               if freeze:
                   for layer in vgg.layers:
                       print (layer.name)
                       if layer.name == '----block5_conv3': continue
                       else: layer.trainable = False
               base_model = tf.keras.Model(vgg.input,vgg.output)
        elif args.model == 'inceptionv3/':
               print('using inceptionv3 imagenet weights for CUB200 dataset')
               inception_v3 = InceptionV3(weights='imagenet',include_top = False,input_shape=(299,299,3))#top needed to get output dimensions at each layer
               freeze=True
               neuron_sum = 0
               if freeze:
                   for layer in inception_v3.layers:
                       # print (layer.name, layer.output.shape)
                       if 'conv' in layer.name: 
                           print (layer.name,'\t\t', layer.output.shape)
                           neuron_sum+=layer.output.shape[-1]
               print("total neurons: ", neuron_sum)
               base_model = tf.keras.Model(inception_v3.input,inception_v3.output)
    elif args.dataset == 'BraTS':
        if args.model == 'VGG16/':
           print('using VGG16 imagenet weights for BraTS dataset')
           vgg = VGG16(weights='imagenet',include_top = False,input_shape=(240,240,3))#top needed to get output dimensions at each layer
           freeze=True
           if freeze:
               for layer in vgg.layers:
                   print (layer.name)
                   if layer.name == '----block5_conv3': continue
                   else: layer.trainable = False
                   
               neuron_sum=0    
               for layer in vgg.layers:
                    # print (layer.name, layer.output.shape)
                    if 'conv' in layer.name: 
                        print (layer.name,'\t\t', layer.output.shape)
                        neuron_sum+=layer.output.shape[-1]
               print("total neurons: ", neuron_sum)                   
           base_model = tf.keras.Model(vgg.input,vgg.layers[-2].output)
    elif args.dataset == 'catsvsdogs':
        print('using imagenet weights for catsvsdogs dataset')
        vgg = VGG16(weights='imagenet',include_top = False,input_shape=(224,224,3))#top needed to get output dimensions at each layer
        freeze=True
        if freeze:
            for layer in vgg.layers:
                layer.trainable = False
        base_model = tf.keras.Model(vgg.input,vgg.layers[-2].output)
    elif args.dataset == 'VOC2010':
        print('using imagenet weights for VOC2010-animals dataset')
        vgg = VGG16(weights='imagenet',include_top = False,input_shape=(224,224,3))#top needed to get output dimensions at each layer
        freeze=True
        if freeze:
            for layer in vgg.layers:
                layer.trainable = False
        base_model = tf.keras.Model(vgg.input,vgg.layers[-2].output)
else:
    base_model = VGG16(weights=None,include_top = False)
    #base_model = MyFunctionalModel()


