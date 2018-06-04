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
from keras.callbacks import ModelCheckpoint

import h5py
import os
import sys
from keras.models import model_from_json


#define params
trn_file = sys.argv[1]
tst_file = sys.argv[2]
json_file = sys.argv[3]
h5_file = sys.argv[4]

nb_classes = 3
nb_kernels = 3
nb_pools = 2

# load training dataset
dataset = numpy.loadtxt(trn_file, delimiter=",", ndmin = 2)
# split into input (X) and output (Y) variables
X = dataset[:,1:401].reshape(len(dataset),1,20,20)
Y = dataset[:,0]

Y = np_utils.to_categorical(Y,nb_classes)
#print X,Y
#nb_classes = Y.shape[1]
#print nb_classes

# load testing dataset
dataset1 = numpy.loadtxt(tst_file, delimiter=",", ndmin = 2)
# split into input (X) and output (Y) variables
X1 = dataset1[:,1:401].reshape(len(dataset1),1,20,20)
Y1 = dataset1[:,0]
true_labels = numpy.asarray(Y1)

Y1 = np_utils.to_categorical(Y1,nb_classes)
# i = 3571
# plt.imshow(X[i,0], interpolation='nearest')
# print('label:', Y[i,:])

def cnn_model():
    model = Sequential()

    #model.add(Dropout(0.2, input_shape = (1,20,20)))
    model.add(ZeroPadding2D((1,1), input_shape = (1,20,20)))
    model.add(Conv2D(32, nb_kernels, nb_kernels, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(32, nb_kernels, nb_kernels, activation='relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering='th'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, nb_kernels, nb_kernels, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(64, nb_kernels, nb_kernels, activation='relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering='th'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, nb_kernels, nb_kernels, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(128, nb_kernels, nb_kernels, activation='relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering='th'))
    
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Conv2D(256, nb_kernels, nb_kernels, activation='relu'))
    # # model.add(ZeroPadding2D((1,1)))
    # # model.add(Conv2D(256, nb_kernels, nb_kernels, activation='relu'))
    # model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering="th"))

    ## add the model on top of the convolutional base
    model.add(Flatten())
#    model.add(Dropout(0.2))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=['accuracy'])

    # Compile model
    return model


model = cnn_model()

filepath = "weights.best.hdf5"
#tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=50, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)

#adadelta
model.fit(X, Y, nb_epoch=200, batch_size=10, class_weight = 'auto', validation_data=(X1,Y1), callbacks=[checkpointer])
scores = model.evaluate(X1, Y1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.load_weights(filepath)

#serialize model to JSON
model_json = model.to_json()
with open(json_file, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(h5_file)
print("Saved model to disk")

