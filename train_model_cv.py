# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:25:26 2018

@author: Khanh Lee
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:08:49 2018

@author: Khanh Lee
"""

import numpy
# fix random seed for reproducibility

seed = 7
numpy.random.seed(seed)
# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten

import h5py
import os
import sys


#define params
trn_file = sys.argv[1]

nb_classes = 3
nb_kernels = 3
nb_pools = 2

# load training dataset
dataset = numpy.loadtxt(trn_file, delimiter=",", ndmin = 2)
# split into input (X) and output (Y) variables
X = dataset[:,1:401].reshape(len(dataset),1,20,20)
Y = dataset[:,0]


def cnn_model():
    model = Sequential()

    model.add(ZeroPadding2D((1,1), input_shape = (1,20,20)))
    model.add(Conv2D(32, nb_kernels, nb_kernels, activation='relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering='th'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, nb_kernels, nb_kernels, activation='relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering='th'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, nb_kernels, nb_kernels, activation='relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering='th'))
    

    ## add the model on top of the convolutional base
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=['accuracy'])

    # Compile model
    return model


# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
    model = cnn_model()   
    # fit the model
    model.fit(X[train], np_utils.to_categorical(Y[train],nb_classes), nb_epoch=200, batch_size=10, verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], np_utils.to_categorical(Y[test],nb_classes), verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    # prediction
    true_labels = numpy.asarray(Y[test])
    predictions = model.predict_classes(X[test])
    print(confusion_matrix(true_labels, predictions))
