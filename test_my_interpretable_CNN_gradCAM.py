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
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%%
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Softmax
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from tqdm import tqdm # to monitor progress
import argparse

from models import MySubClassModel

#%%
parser = argparse.ArgumentParser(description='Interpretable CNN')
parser.add_argument('--interpretable',default = True)

#global args 
args = parser.parse_args()
#%%
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#%%
imgDataGen = ImageDataGenerator(rescale = 1./255)

train_gen = imgDataGen.flow(x_train, y_train, batch_size = 32,shuffle= False)
test_gen  = imgDataGen.flow(x_test, y_test, batch_size = 32,shuffle= False)

#%%
def MyFunctionalModel():
    inputs = tf.keras.Input(shape = (28,28,1))
  
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu',strides=(1,1))(x)

    return tf.keras.Model(inputs, x)

#%%
base_model = MyFunctionalModel()

model = MySubClassModel(num_classes=10, base_model=base_model, args=args)
#model(tf.zeros((1,28,28,1)))
model.build(input_shape = (1,28,28,1))

model.summary()

if args.interpretable:
    model.load_weights('saved_model_weights_interpretable')
else:
    model.load_weights('saved_model_weights_standard')

#%%
x_batch_test, y_batch_test = next(test_gen)
#%%
from tf_explain_modified.core.grad_cam import GradCAM
#Instantiation of the explainer
explainer = GradCAM()
#%%
img_ind = np.random.randint(0,32)

#pred_probs,fmaps_x1 = model.predict(np.expand_dims(x_batch_test[img_ind],0))#without eager
pred_probs,fmaps_x1 = model(np.expand_dims(x_batch_test[img_ind],0))#with eager


y_gt = y_batch_test[img_ind]

plt.imshow(x_batch_test[img_ind].squeeze(),cmap='gray')
plt.show()
print('predicted: ',np.argmax(pred_probs), ' with prob: ',np.max(pred_probs)*100,'%')
print ('actual: ', np.argmax(y_gt))

plt.imshow(fmaps_x1[0,:,:,0],cmap='gray')
plt.show()

output = explainer.explain((np.expand_dims(x_batch_test[img_ind],0),None),model,np.argmax(y_batch_test[img_ind]))

plt.imshow(output,cmap='gray')
plt.show()
#%%
fig, axs = plt.subplots(8,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
#fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

for i in range(32):

    axs[i].imshow(fmaps_x1[0,:,:,i],cmap='gray')#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
    axs[i].axis('off')
    
plt.show()
#%%