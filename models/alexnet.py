import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization

from PIL import Image


alexnet_model = Sequential()

# Layer 1
alexnet_model.add(Convolution2D(96, 11, 11, input_shape =(224,224,3), border_mode='same'))
alexnet_model.add(Activation('relu'))
alexnet_model.add(MaxPooling2D(pool_size=(2, 2)))


# Layer 2
alexnet_model.add(Convolution2D(256, 5, 5, border_mode='same'))
alexnet_model.add(Activation('relu'))
alexnet_model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
alexnet_model.add(ZeroPadding2D((1,1)))
alexnet_model.add(Convolution2D(512, 3, 3, border_mode='same'))
alexnet_model.add(Activation('relu'))

# Layer 4
alexnet_model.add(ZeroPadding2D((1,1)))
alexnet_model.add(Convolution2D(1024, 3, 3, border_mode='same'))
alexnet_model.add(Activation('relu'))

# Layer 5
alexnet_model.add(ZeroPadding2D((1,1)))
alexnet_model.add(Convolution2D(1024, 3, 3, border_mode='same'))
alexnet_model.add(Activation('relu'))
alexnet_model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 6
alexnet_model.add(Flatten())
alexnet_model.add(Dense(3072, init='glorot_normal'))
alexnet_model.add(Activation('relu'))
alexnet_model.add(Dropout(0.5))

# Layer 7
alexnet_model.add(Dense(4096, init='glorot_normal'))
alexnet_model.add(Activation('relu'))
alexnet_model.add(Dropout(0.5))

# Layer 8
alexnet_model.add(Dense(2, init='glorot_normal'))
alexnet_model.add(Activation('softmax'))

alexnet_model.summary()