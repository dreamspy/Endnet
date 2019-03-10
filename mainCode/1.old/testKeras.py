fileName = '4PpKk.hdf5'
dsNameStates = '4PpKk-fullStateBool'
dsNameWDLs = '4PpKk-Wdl'
convertStates = False
# fileName = 'AllStates_intVec.hdf5'
nPi = 4
nPa = 2
nWPa = 1
input_shape = (4,8,8)
batch_size = 256
epochs = 300
dataDiv = 1
num_classes = 5

multiGPU = False
whichGPU = 0



# from __future__ import print_function
import keras
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
# import pandas as pd
#from sklearn.model_selection import train_test_split
import sys
#import h5py
#import numpy as np



model = Sequential()
filterDivider = 2
l1filters = 64//filterDivider
l2filters = 96//filterDivider
l3filters = 256//filterDivider
print(l1filters)
print(l2filters)
print(l3filters)

model.add(Conv2D(l1filters, kernel_size=(3, 3),
                 padding='valid',
                 activation='relu',
                 data_format = "channels_first",
#                  kernel_initializer =
                 input_shape=input_shape))
model.add(Conv2D(l2filters, kernel_size=(3, 3),
                 padding='valid',
                 activation='relu'))
model.add(Conv2D(l3filters, kernel_size=(3, 3),
                 padding='valid',
                 activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
# model = keras.utils.multi_gpu_model(model, gpus=2)
model.summary()
model.save_weights('tempWeights.hdf5')

