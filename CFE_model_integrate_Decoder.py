# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 21:04:33 2022

@author: Ali
"""

#%%
import tensorflow as tf
print('TF version: ', tf.__version__)

from tensorflow.keras.layers import Dense,ZeroPadding2D, Conv2D, MaxPooling2D, Flatten,Softmax, GlobalAveragePooling2D, Dropout, UpSampling2D,Conv2DTranspose
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2 as cv
import pathlib

from share_model import model as classifier
from share_model import restore_original_image_from_array

