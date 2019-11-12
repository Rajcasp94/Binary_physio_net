# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:01:47 2019

@author: rajde
"""
#import numpy as np
#from keras.utils import to_categorical
#import matplotlib.pyplot as plt
#import pandas as pd
#import matplotlib.image as img
#from skimage import io
#from scipy.misc import imshow
#from PIL import Image
#from scipy.misc import toimage
#from scipy.misc import imresize
#from keras.models import Sequential,Model
#from keras.layers import Dense,Activation,Dropout,Flatten,Input,Conv2D,TimeDistributed,LSTM
#from keras.layers import InputLayer
#
#import tensorflow as tf



from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.layers import LSTM, Conv1D, MaxPooling1D, BatchNormalization,Flatten,Reshape,Dropout
from keras.utils import plot_model

import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'


model_signal1_in = Input(shape=(1,5000))
model_signal1_layer_1=Conv1D(filters=100,kernel_size=1,strides=50,activation='sigmoid')(model_signal1_in)
model_signal1_layer_2=MaxPooling1D(1,strides=5)(model_signal1_layer_1)
model_signal1_layer_3=BatchNormalization()(model_signal1_layer_2)
model_signal1_layer_4=Conv1D(filters=100,kernel_size=1,strides=50,activation='sigmoid')(model_signal1_layer_3)
model_signal1_layer_5=MaxPooling1D(1,strides=5)(model_signal1_layer_4)
model_signal1_layer_6=Dropout(0.2)(model_signal1_layer_5)
model_signal1_layer_7=Dense(100,activation='sigmoid')(model_signal1_layer_6)
model_signal1_layer_8=Dropout(0.2)(model_signal1_layer_7)
model_signal1_layer_9=Reshape((100, 1))(model_signal1_layer_8)
model_signal1_layer_10=LSTM(100,activation='sigmoid',return_sequences=True)(model_signal1_layer_9)
model_signal1_layer_11=Dropout(0.2)(model_signal1_layer_10)
model_signal1_layer_12=LSTM(200,activation='sigmoid')(model_signal1_layer_11)
model_signal1_layer_13=Dropout(0.2)(model_signal1_layer_12)
model_signal1_layer_14=BatchNormalization()(model_signal1_layer_13)
model_signal1_out = Dense(300, activation='sigmoid')(model_signal1_layer_14)
model_signal1 = Model(model_signal1_in,
                  model_signal1_layer_1,
                  model_signal1_layer_2,
                  model_signal1_layer_3,
                  model_signal1_layer_4,
                  model_signal1_layer_5,
                  model_signal1_layer_6,
                  model_signal1_layer_7,
                  model_signal1_layer_10,
                  model_signal1_layer_11,
                  model_signal1_layer_12,
                  model_signal1_layer_13,
                  model_signal1_layer_14,
                  model_signal1_out)

model_signal2_in = Input(shape=(1,5000))
model_signal2_layer_1=Conv1D(filters=100,kernel_size=1,strides=50,activation='sigmoid')(model_signal2_in)
model_signal2_layer_2=MaxPooling1D(1,strides=5)(model_signal2_layer_1)
model_signal2_layer_3=BatchNormalization()(model_signal2_layer_2)
model_signal2_layer_4=Conv1D(filters=100,kernel_size=1,strides=50,activation='sigmoid')(model_signal2_layer_3)
model_signal2_layer_5=MaxPooling1D(1,strides=5)(model_signal2_layer_4)
model_signal2_layer_6=Dropout(0.2)(model_signal2_layer_5)
model_signal2_layer_7=Dense(100,activation='sigmoid')(model_signal2_layer_6)
model_signal2_layer_8=Dropout(0.2)(model_signal2_layer_7)
model_signal2_layer_9=Reshape((100, 1))(model_signal2_layer_8)
model_signal2_layer_10=LSTM(100,activation='sigmoid',return_sequences=True)(model_signal2_layer_9)
model_signal2_layer_11=Dropout(0.2)(model_signal2_layer_10)
model_signal2_layer_12=LSTM(200,activation='sigmoid')(model_signal2_layer_11)
model_signal2_layer_13=Dropout(0.2)(model_signal2_layer_12)
model_signal2_layer_14=BatchNormalization()(model_signal2_layer_13)
model_signal2_out = Dense(300, activation='sigmoid')(model_signal2_layer_14)
model_signal2 = Model(model_signal2_in, 
                  model_signal2_layer_1,
                  model_signal2_layer_2,
                  model_signal2_layer_3,
                  model_signal2_layer_4,
                  model_signal2_layer_5,
                  model_signal2_layer_6,
                  model_signal2_layer_7,
                  model_signal2_layer_10,
                  model_signal2_layer_11,
                  model_signal2_layer_12,
                  model_signal2_layer_13,
                  model_signal2_layer_14,
                  model_signal2_out)


concatenated = concatenate([model_signal1_out, model_signal2_out])
concantened_layer_1=BatchNormalization()(concatenated)
concantened_layer_2=Reshape((600, 1))(concantened_layer_1)
concantened_layer_3=LSTM(50,activation='sigmoid',return_sequences=True)(concantened_layer_2)
concantened_layer_4=LSTM(100,activation='sigmoid')(concantened_layer_3)
concantened_layer_5=BatchNormalization()(concantened_layer_4)
concantened_layer_6=Dense(300, activation='sigmoid')(concantened_layer_5)
out = Dense(1, activation='softmax', name='output_layer')(concantened_layer_6)

merged_model = Model([model_signal1_in, model_signal2_in], out)



plot_model(merged_model, to_file='model.png')
print(merged_model)





