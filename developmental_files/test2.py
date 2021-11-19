# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:43:27 2020

@author: Ali
"""

#%%
import tensorflow as tf
print('TF version: ', tf.__version__)

#%% fix for issue: cuDNN failed to initialize
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

#from models import MySubClassModel
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
class MySubClassModel(Model):

  def __init__(self,num_classes=10):
    super(MySubClassModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    
    #it should be a pre-trained model according to paper
    self.base_model = MyFunctionalModel()# assume: without top layers (no classification layers)
    
    output_shape = self.base_model.get_output_shape_at(0)
    self.n = output_shape[1] #target conv layer output size
    self.k = output_shape[3] #number of output filters
    
    self.tao = (0.5)/np.power(self.n,2) #tao is positive constant
    self.beta = 4 #beta is postive constant
    
    #create templates corresponding to each feature map (i.e output shape, (none, 11,11,32) --> means 32 templates of size 11x11)
    self.t_p = self.create_templates() #positive templates for each filter; shape: (kxnxnx(nxn))
    # single negative template: it should be appended with the positive templates, to be used during backpropagation if Image does not belong to target class
    self.t_n = tf.convert_to_tensor(np.ones((self.n,self.n))*-self.tao)



    # Define your layers here.
    self.conv1 = Conv2D(32, (3, 3), activation='relu',padding='same')
    #self.pool1 = MaxPool2D(pool_size=(2, 2))
    self.flatten = Flatten()
    self.dense_1 = Dense(32, activation='relu')
    self.dense_2 = Dense(10, activation='softmax')


  def create_templates(self):
    t_p = np.zeros((self.k,self.n,self.n,self.n,self.n))
    for k in range(self.k):
        for i in range(self.n):
            for j in range(self.n):
                t_p[k,i,j,:,:] = self.create_template_single(mu=np.array([i,j]))
    #plt.imshow(t_p[0,0,0,:,:],cmap='gray')
    t_p_tensor = tf.convert_to_tensor(t_p)

    #t_n = np.ones((self.n,self.n))*-self.tao
    
    return t_p_tensor#,t_n
  
  def create_template_single(self,mu):
    t_p = np.zeros((self.n,self.n))
    for i in range(self.n):
        for j in range(self.n):
            t_p[i,j] = self.tao*max(1-self.beta*(np.linalg.norm(np.array([i, j]) - mu, ord=1)/self.n),-1)    
    return t_p
    
  def compute_loss(self, x):
      #implement the -MI loss 
      return 1
      
      
  def get_masked_output(self,x):
    # find index of strongest activation:mu=[i,j]
    # find masked output of each filter with corresponding template
    x=x*1
    return x

  def call(self, inputs):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).

    x = self.base_model(inputs)
    x1 = self.get_masked_output(x)
    x = self.conv1(x1)
    x2 = self.get_masked_output(x)
    x = self.flatten(x2)
    x = self.dense_1(x)
    x = self.dense_2(x)
    # out = np.array([x1,x2])

    return x, zip(x1,x2)


def MyFunctionalModel():
    inputs = tf.keras.Input(shape = (28,28,1))
  
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu',strides=(1,1), name='target_conv')(x)

    return tf.keras.Model(inputs, x)

#%%
model = MySubClassModel(num_classes=10)
#model(tf.zeros((1,28,28,1)))
model.build(input_shape = (1,28,28,1))
# The compile step specifies the training configuration.
#%%
#model = MyFunctionalModel()

model.summary()

#%% Trains for 5 epochs.
# model.compile(optimizer=RMSprop(0.001),
#               loss=[categorical_crossentropy, None],loss_weights=[1,0],#, my_loss],
#                 metrics=['accuracy'])
# model.fit(train_gen, epochs=5, verbose=1, callbacks=None, validation_data=None, shuffle=True)

#%%
#dataset = tf.data.Dataset.from_generator(lambda:train_gen,(tf.float32, tf.float32)) 



#%% manual training loop
#with or without logits is making a difference in accuracy and loss when compared to model.fit
loss_fn = tf.keras.losses.CategoricalCrossentropy()#from_logits=True
optimizer = RMSprop(0.001)

train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

#%%
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions,fts = model(images, training=True)
    loss_value = loss_fn(labels, predictions)
  gradients = tape.gradient(loss_value, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss_metric(loss_value)
  train_acc_metric(labels, predictions)
  #return loss_value
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions,_ = model(images, training=False)
  loss_value = loss_fn(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
  
import math
epochs = 5
batches=math.ceil(train_gen.n/train_gen.batch_size)

# Iterate over the batches of the dataset.
#for step, (x_batch_train, y_batch_train) in enumerate(train_gen):

 
#%%
for epoch in range(epochs):
  print('Start of epoch %d' % (epoch,))
  
  #for step,(x_batch_train, y_batch_train) in enumerate(dataset):
  for step in range(batches):
    x_batch_train, y_batch_train = next(train_gen)
    #t_x = np.zeros((32,11,11,32))


    #This method not executing eagerly::
    #loss_value = model.train_on_batch(x_batch_train,[y_batch_train])#, t_x])
    #loss_value = loss_value[1]
    #:::::::::::::::::;
    
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables autodifferentiation.
    train_step(x_batch_train, y_batch_train)
    
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


    # Log every 200 batches.
    if step % 200 == 0:
        print('Training loss (for one batch) at step %s: ' % (step))#, float(loss_value)))
        #print('Seen so far: %s samples' % ((step + 1) * 32))

   # Display metrics at the end of each epoch.
  train_acc = train_acc_metric.result()
  train_loss = train_loss_metric.result()

  print('Training acc over epoch: %s' % (float(train_acc),))
  print('Training loss over epoch: %s' % (float(train_loss),))

  # Reset training metrics at the end of each epoch
  train_acc_metric.reset_states()
  train_loss_metric.reset_states()



#%%
#test_scores = model.evaluate(test_gen, verbose=1)
#print('Test loss:', test_scores[0])
#print('Test accuracy:', test_scores[1])


#%% manual test
pred_probs,_= model.predict(test_gen,verbose=1)

pred_classes = np.argmax(pred_probs,1)
actual_classes = np.argmax(test_gen.y,1)
print(confusion_matrix(actual_classes,pred_classes))
print(classification_report(actual_classes,pred_classes))
#print('Test loss:', test_scores[0])
#print('Test accuracy:', test_scores[1])
#%%
x_batch_test,y_batch_test = next(test_gen)
#%%
img_ind = np.random.randint(0,32)
pred_probs,fmaps_x1 = model.predict(np.expand_dims(x_batch_test[img_ind],0))

y_gt = y_batch_test[img_ind]

plt.imshow(x_batch_test[img_ind].squeeze(),cmap='gray')
plt.show()
print('predicted: ',np.argmax(pred_probs), ' with prob: ',np.max(pred_probs)*100,'%')
print ('actual: ', np.argmax(y_gt))

plt.imshow(fmaps_x1[0,:,:,0],cmap='gray')
plt.show()



#%%
model.save_weights('saved_model_weights')










