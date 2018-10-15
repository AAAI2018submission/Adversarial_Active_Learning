# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:17:56 2017

@author: mducoffe
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from regulizer import get_regularizer

def build_model_AlexNet(img_size, reg_name, regulizer, nb_classes):
    
    nb_pool = 2
    model = Sequential()
     
    nb_channel, img_rows, img_cols = img_size
    	#layer 1
    model.add(Conv2D(96, (11, 11), padding='same', input_shape = (nb_channel, img_rows, img_cols), 
                     data_format='channels_first', kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    #layer 2
    model.add(Conv2D(256, (5, 5), padding='same', data_format='channels_first', kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    #layer 3
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), padding='same', data_format='channels_first', kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Activation('relu'))
    
    #layer 4
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(1024, (3, 3), padding='same', data_format='channels_first', kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Activation('relu'))
    
    #layer 5
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(1024, (3, 3), padding='same', data_format='channels_first', kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    #layer 6
    model.add(Flatten())
    model.add(Dense(3072, kernel_initializer="glorot_normal", kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    #layer 7
    model.add(Dense(4096, kernel_initializer="glorot_normal", kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    #layer 8
    model.add(Dense(nb_classes, kernel_initializer="glorot_normal", kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["acc"])
    
    return model;
    
def build_model_VGG8(img_size, reg_name, regulizer, nb_classes):
    
    nb_conv = 3
    nb_pool = 2
    nb_channel, img_rows, img_cols = img_size

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(nb_channel,img_rows,img_cols)))
    model.add(Conv2D(64, (nb_conv, nb_conv), activation='relu', data_format='channels_first',kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (nb_conv, nb_conv), activation='relu', data_format='channels_first',kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (nb_conv, nb_conv), activation='relu', data_format='channels_first',kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (nb_conv, nb_conv), activation='relu', data_format='channels_first',kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (nb_conv, nb_conv), activation='relu', data_format='channels_first',kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (nb_conv, nb_conv), activation='relu', data_format='channels_first',kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu',kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes,kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Activation('softmax'))
    
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["acc"])
     
    return model
     
def build_model_LeNet5(img_size, reg_name, regulizer, nb_classes):

    nb_pool = 2
    nb_channel, img_rows, img_cols = img_size
    model = Sequential()
    
    model.add(Conv2D(6, (5, 5), padding='valid', 
                     input_shape = (nb_channel, img_rows, img_cols), 
                     data_format='channels_first', kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(16, (5, 5), padding='valid', data_format='channels_first', kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(120, (1, 1), padding='valid', data_format='channels_first',kernel_regularizer=get_regularizer(reg_name, regulizer)))
    
    model.add(Flatten())
    model.add(Dense(84, activation='relu', kernel_constraint=None, kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Dense(nb_classes, kernel_constraint=None, kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, kernel_constraint=None, kernel_regularizer=get_regularizer(reg_name, regulizer)))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["acc"])
     
    return model
     
def build_model_func(network_archi, img_size=(1,28,28, 10)):
    """
    if not(reg_name is None):    
        reg_name = reg_name.lower()
    """
    reg_name = None
    network_archi = network_archi.lower()
    regulizer=0
    num_classes = img_size[3]
    img_size = (img_size[0], img_size[1], img_size[2])
    model = None
    assert (network_archi in ['vgg8', 'lenet5', 'alexnet']), ('unknown architecture', network_archi)
    if network_archi == 'vgg8':
        model = build_model_VGG8(img_size, reg_name=reg_name, regulizer=regulizer, nb_classes=num_classes)
    if network_archi == 'lenet5':
        model = build_model_LeNet5(img_size, reg_name=reg_name, regulizer=regulizer, nb_classes=num_classes)
    if network_archi == 'alexnet':
        model = build_model_AlexNet(img_size, reg_name=reg_name, regulizer=regulizer, nb_classes=num_classes)
        
    return model
