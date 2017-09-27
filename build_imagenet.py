#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:36:03 2017

@author: mducoffe

Tiny Imagenet
"""
import sys
#sys.path.append('./snapshot')
#import json
import numpy as np
import sklearn.metrics as metrics
import argparse
import keras
import keras.utils.np_utils as kutils
from keras.datasets import cifar10
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from adversarial_active_criterion import Adversarial_DeepFool
#from snapshot import SnapshotCallbackBuilder
import csv
from contextlib import closing
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D
from scipy import misc
import pickle as pkl
#%%

def build_pkl_data():
    
    repository='./imagenet/tiny-imagenet-200'
    
    repository_train = os.path.join(repository, 'train')
    listdir_train = os.listdir(repository_train)
    dico_labels = dict([(listdir_train[i], i) for i in range(len(listdir_train))])
    
    
    x_train = []
    y_train = []
    for repo_class in listdir_train:
        data_repo = os.path.join(repository_train, os.path.join(repo_class, 'images'))
        images = os.listdir(data_repo)
        
        for image in images:
            tmp = misc.imread(os.path.join(data_repo, image))
            if tmp.ndim!=3:
                continue
            x_train.append(tmp)
            y_train.append(dico_labels[repo_class])
    x_train = np.array(x_train)
    y_train = np.array(y_train)   
    
    # load validation data
    repository_val = os.path.join(repository, 'val')

    with closing(open(os.path.join(repository_val, 'val_annotations.txt'))) as f:
        content = f.readlines()
        
    dico_val = {}
    for line in content:
        data = line.split('\t')
        dico_val[data[0]] = dico_labels[data[1]]
    
    x_test = []; y_test = []
    
    repo_val = os.path.join(repository_val, 'images')
    images_val = os.listdir(repo_val)
    for image in images_val:
        tmp = misc.imread(os.path.join(repo_val, image))
        if tmp.ndim!=3:
            continue
        x_test.append(tmp)
        y_test.append(dico_val[image])
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return [(x_train, y_train), (x_test, y_test)]

def build_data(num_sample=100, img_rows=32, img_cols=32):
    with closing(open('imagenet_data.pkl', 'rb')) as f:
        train, test = pkl.load(f)
    
    x_train, y_train = train
    x_test, y_test = test
    
    x_train = 1.*x_train/255
    x_train = x_train.astype('float32')
    x_train /= 255.0
    x_test = x_test.astype('float32')
    x_test /= 255.0
    
    y_train = kutils.to_categorical(y_train)
    y_test = kutils.to_categorical(y_test)
    N = len(x_train)
    index = np.random.permutation(N)
    x_train = x_train[index]
    y_train = y_train[index]
    
    #x_L = x_train[:num_sample]; y_L = y_train[:num_sample]
    x_L = x_train; y_L = y_train
    x_U = x_train[num_sample:]; y_U = y_train[num_sample:]


    return (x_L, y_L), (x_U, y_U), (x_test, y_test)
    
#%%
def build_model(num_classes=200):

    img_rows, img_cols = 64, 64
    
    if K.image_dim_ordering() == "th":
        input_shape = (3, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 3)
        
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3,3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.01, decay=1e-6)
    print('KIKOU')
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
    return model

#%%
def active_training(labelled_data,
                    batch_size=64, epochs=100, repeat=1):
    
    x_L, y_L = labelled_data 
    
    # split into train and validation
    index = np.random.permutation(len(y_L))
    N = len(index)
    x_L = x_L[index]; y_L = y_L[index]
    
    
    
    # train and valid generator
    generator_train = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=5./32,
                                   height_shift_range=5./32,
                                   horizontal_flip=True)

    generator_train.fit(x_L, seed=0, augment=True)
    print('A')
    batch_train = min(batch_size, len(x_L))
    steps_per_epoch_train = int(N/batch_train)
    tmp = generator_train.flow(x_L, y_L, batch_size=batch_size)
    values = [tmp.next() for i in range(steps_per_epoch_train)]
    x_L = np.concatenate([value[0] for value in values])
    y_L = np.concatenate([value[1] for value in values])
    
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    
    best_model = None
    best_loss = np.inf
    for i in range(repeat):
        print('B')
        model = build_model()
        print('C')
        model.fit(x_L, y_L, 
             batch_size=batch_train, epochs=epochs,
             callbacks=[earlyStopping],
             shuffle=True,
             validation_split=0.2,
             verbose=1)

        loss, acc = model.evaluate(x_L, y_L, verbose=0)
        if loss < best_loss:
            best_loss = loss;
            best_model = model

    return best_model

def evaluate(model, percentage, test_data, nb_exp, repo, filename):
    x_test, y_test = test_data
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    
    with closing(open(os.path.join(repo, filename), 'a')) as csvfile:
        # TO DO
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([str(nb_exp), str(percentage), str(loss), str(acc)])
        
#%%
def active_learning(num_sample=1000, percentage=0.99, 
                    active_method='adversarial', nb_exp=0, repo='test', filename='test.csv'):
    
    batch_size = 32
    labelled_data, unlabelled_data, test_data = build_data(num_sample=num_sample)
    #percentage_data = 1.*len(labelled_data[0])/len(unlabelled_data[0])
    percentage_data = 0.1
    # load data
    i=0
    while( percentage_data<=percentage):

        i+=1
        model = active_training(labelled_data, batch_size=batch_size)
        """
        query, unlabelled_data = active_selection(model=model, 
                                                  unlabelled_data=unlabelled_data, 
                                                  active_method=active_method, 
                                                  nb_data=batch_size)
        """
        evaluate(model, percentage_data, test_data, nb_exp, repo, filename)
        print('SUCEED')
        return
        # add query to the labelled set
        labelled_data_0 = np.concatenate((labelled_data[0], query[0]), axis=0)
        labelled_data_1 = np.concatenate((labelled_data[1], query[1]), axis=0)
        labelled_data = (labelled_data_0, labelled_data_1)
        #update percentage_data
        percentage_data = 1.*len(labelled_data[0])/len(unlabelled_data[0])
        

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Active Learning on MNIST')

    parser.add_argument('--active_method', type=str, default='random', help='active learning selection')
    parser.add_argument('--id_experiment', type=int, default=3, help='id number of experiment')
    parser.add_argument('--repo', type=str, default='.', help='repository for log')
    parser.add_argument('--filename', type=str, default='cifar_random_v0', help='csv filename')
    args = parser.parse_args()

    
    active_method=args.active_method
    nb_exp = args.id_experiment
    repo=args.repo
    filename=args.filename
    if filename.split('.')[-1]!='csv':
        filename+='.csv'
    
    active_learning(active_method=active_method,
                    nb_exp=nb_exp,
                    repo=repo,
                    filename=filename)



