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
    x = Conv2D(32, (3, 3), activation='relu',strides=(1,1), name='target_conv')(x)

    return tf.keras.Model(inputs, x)

#%%
base_model = MyFunctionalModel()

model = MySubClassModel(num_classes=10, base_model=base_model, args=args)
#model(tf.zeros((1,28,28,1)))
model.build(input_shape = (1,28,28,1))

model.summary()


model.load_weights('saved_model_weights')

#%% manual testing loop
"""with or without logits is making a difference in accuracy and loss when compared to model.fit
#why? because i have used softmax as final output layer so the network outputs probabilities instead of logits... therefore from_logits must if False in this case.
#   if i want to apply softmax after getting the output from the network, then i can use from_logits=True to compute the loss """

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
optimizer = RMSprop(0.001)

test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
test_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

#%%
y_pred = []
y_true = []
"""tf.function constructs a callable that executes a TensorFlow graph (tf.Graph) created by trace-compiling the TensorFlow operations in func, effectively executing func as a TensorFlow graph.
#uncomment following line for faster training but it becomes non-debugable and doesnt execute eagerly"""
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions,_ = model(images, training=False)
  loss_value = loss_fn(labels, predictions)

  test_loss_metric(loss_value)
  test_acc_metric(labels, predictions)
  return predictions
  #y_pred.append(predictions)
  #y_true.append(labels)

 
#%%
# Iterate over the batches of the dataset.
#for step, (x_batch_train, y_batch_train) in enumerate(train_gen):

import math
batches=math.ceil(test_gen.n/test_gen.batch_size)

for epoch in range(1):
  #print('Start of epoch %d' % (epoch,))
  
  #for step,(x_batch_train, y_batch_train) in enumerate(dataset):
  with tqdm(total=batches) as progBar:
      for step in range(batches):
        x_batch_test, y_batch_test = next(test_gen)
        #t_x = np.zeros((32,11,11,32))
    
    
        #This method not executing eagerly::
        #loss_value = model.train_on_batch(x_batch_train,[y_batch_train])#, t_x])
        #loss_value = loss_value[1]
        #:::::::::::::::::;
        
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        
        #using tf.function method for faster training
        preds = test_step(x_batch_test, y_batch_test) 
        y_pred.append(preds)
        y_true.append(y_batch_test)
        # with tf.GradientTape() as tape:
    
        #   # Run the forward pass of the layer.
        #   # The operations that the layer applies
        #   # to its inputs are going to be recorded
        #   # on the GradientTape.
        #   logits,_ = model(x_batch_train, training=True)  # Logits for this minibatch
    
        #   # Compute the loss value for this minibatch.
        #   loss_value = loss_fn(y_batch_train, logits)
    
        # # Use the gradient tape to automatically retrieve
        # # the gradients of the trainable variables with respect to the loss.
        # grads = tape.gradient(loss_value, model.trainable_weights)
    
        # # Run one step of gradient descent by updating
        # # the value of the variables to minimize the loss.
        # optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
        # # Update training metric.
        # train_acc_metric(y_batch_train, logits)
        # train_loss_metric(loss_value)

    
    
        # Log every 200 batches.
        #if step % 200 == 0:
            #print('Training loss (for one batch) at step %s: ' % (step))#, float(loss_value)))
            #print('Seen so far: %s samples' % ((step + 1) * 32))
        progBar.set_description('epoch %d' % (epoch))
        progBar.set_postfix(loss=test_loss_metric.result().numpy(), acc=test_acc_metric.result().numpy())
        progBar.update()

    
       # Display metrics at the end of each epoch.
      test_acc = test_acc_metric.result()
      test_loss = test_loss_metric.result()
    
      #print('\nTest acc: %s' % (float(test_acc),))
      #print('Test loss: %s' % (float(test_loss),))
    
      # Reset training metrics at the end of each epoch
      #test_acc_metric.reset_states()
      #test_loss_metric.reset_states()
#%%
y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)

print(confusion_matrix(y_true,y_pred))
print(classification_report(y_true,y_pred))
#%%
x_batch_test, y_batch_test = next(test_gen)
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


#%%
#plt.subplot(8,4)