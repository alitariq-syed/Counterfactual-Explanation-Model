# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 09:26:10 2023

@author: Ali
"""
from tensorflow.keras.layers import Dense,ZeroPadding2D, Reshape,Conv2D, MaxPool2D, Flatten,Softmax, GlobalAveragePooling2D, Dropout, UpSampling2D,Conv2DTranspose
from tensorflow import keras


# Convolutional Encoder
input_img = keras.Input(shape=(28,28,1))
x = Conv2D(64,(3,3),padding="same", activation="relu")(input_img)
x = Conv2D(64,(3,3),padding="same", activation="relu")(x)
x = MaxPool2D((2,2))(x)
x = Conv2D(128, (3,3), padding="same", activation="relu")(x)
x = Conv2D(128, (3,3), padding="same", activation="relu")(x)
x = MaxPool2D((2,2))(x)
x = Conv2D(256, (3,3), padding="same", activation="relu")(x)
x = Conv2D(256, (3,3), padding="same", activation="relu")(x)
x = Conv2D(256, (3,3), padding="same", activation="relu")(x)
x = MaxPool2D((2,2))(x)
x = Conv2D(512, (3,3), padding="same", activation="relu")(x)
x = Conv2D(512, (3,3), padding="same", activation="relu")(x)
x = Conv2D(512, (3,3), padding="same", activation="relu")(x)


encoded = MaxPool2D((2,2),name='maxpool')(x)

# Classification
mean_fmap  = GlobalAveragePooling2D()(encoded)
dropout    = Dropout(0.5)(mean_fmap)
out        = Dense(10,activation='softmax', name='classification')(dropout)

classifier = keras.Model(inputs = input_img, outputs = out, name='Classifier')