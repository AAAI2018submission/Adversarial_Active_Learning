#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:39:29 2017

@author: train mnist on a subset of mnist
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:38:50 2017

@author: tests
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras.utils.np_utils as kutils
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from contextlib import closing
from adversarial_active_criterion import Adversarial_DeepFool
from keras import optimizers
import csv
from annexe_QBDC import disagreement_coeff


def build_data(num_sample=10, img_rows=28, img_cols=28):
        # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_dim_ordering() == "th":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        #input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        #input_shape = (img_rows, img_cols, 1)
        
        
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # split label, unlabelled and test
    
    N = len(x_train)
    index = np.random.permutation(N)
    x_train = x_train[index]
    y_train = kutils.to_categorical(y_train[index])
    
    x_L = x_train[:num_sample]; y_L = y_train[:num_sample]
    
    x_U = x_train[num_sample:]; y_U = y_train[num_sample:]
    
    return (x_L, y_L), (x_U, y_U), (x_test, kutils.to_categorical(y_test))
    
def build_model_MNIST(num_classes=10, img_rows=28, img_cols=28):

    # the data, shuffled and split between train and test sets
    if K.image_dim_ordering() == "th":
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)


    model = Sequential()
    #model.add(Reshape(input_shape, input_shape=(np.prod(input_shape),)))
    model.add(Conv2D(32, 3, 3,
                     activation='relu',
                     input_shape=input_shape))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='linear'))
    model.add(Activation('softmax'))
    
    sgd = optimizers.SGD(lr=1e-1)
    model.compile(loss="categorical_crossentropy",
              #optimizer=keras.optimizers.Adadelta(),
              optimizer=sgd,
              metrics=['accuracy'])
    
    #model.fit(x_train, y_train, epochs=3, validation_split=0.2, verbose=0)
    return model

def active_training(labelled_data, 
                    batch_size=32, epochs=100, repeat=5):
    
    #init model
    best_model = None
    best_loss = np.inf
    for i in range(repeat):
        model = build_model_MNIST()
        
        # labelled data
        x_L, y_L = labelled_data
        #shuffle
        index = np.random.permutation(len(x_L))
        x_L = x_L[index]
        y_L= y_L[index]
        # train model on labelled data
        
        #model.fit(x_train, y_train, epochs=3, validation_split=0.2, verbose=0)
        # here do cross validation
        earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        model.fit(x_L, y_L, epochs=100, 
                  batch_size=batch_size, 
                  shuffle=True,
                  validation_split=0.2, 
                  callbacks=[earlyStopping],
                  verbose=0)
        loss, acc = model.evaluate(x_L, y_L, verbose=0)
        if loss < best_loss:
            best_loss = loss;
            best_model = model

    return best_model

def evaluate(model, percentage, test_data, nb_exp, repo, filename):
    x_test, y_test = test_data
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    # print
    # do with csv file
    with closing(open(os.path.join(repo, filename), 'a')) as csvfile:
        # TO DO
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([str(nb_exp), str(percentage), str(loss), str(acc)])
        
def active_selection(model, unlabelled_data, nb_data, active_method, threshold=0.5):
    assert active_method in ['uncertainty', 'adversarial', 'egl', 'qbc', 'random'], ('Unknown active criterion %s', active_method)
    if active_method=='uncertainty':
        query, unlabelled_data = uncertainty_selection(model, unlabelled_data, nb_data, threshold)
    if active_method=='random':
        query, unlabelled_data = random_selection(unlabelled_data, nb_data)
    if active_method=='egl':
        query, unlabelled_data = egl_selection(model, unlabelled_data, nb_data)
    if active_method=='qbc':
        query, unlabelled_data = qbc_selection(model, unlabelled_data, nb_data)
    if active_method=='adversarial':
        query, unlabelled_data = adversarial_selection(model, unlabelled_data, nb_data)
        
        
    return query, unlabelled_data


def random_selection(unlabelled_data, nb_data):
    index = np.random.permutation(len(unlabelled_data[0]))
    index_query = index[:nb_data]
    index_unlabelled = index[nb_data:]
    
    return (unlabelled_data[0][index_query], unlabelled_data[1][index_query]), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])

"""
def uncertainty_selection(model, unlabelled_data, nb_data):
    
    preds = model.predict(unlabelled_data[0])
    log_pred = -np.log(preds)
    entropy = np.sum(preds*log_pred, axis=1)
    # do entropy
    index = np.argsort(entropy)[::-1]
    
    index_query = index[:nb_data]
    index_unlabelled = index[nb_data:]
    
    return (unlabelled_data[0][index_query], unlabelled_data[1][index_query]), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])
"""
def uncertainty_selection(model, unlabelled_data, nb_data, threshold):
    
    labelled_data, unlabelled_data = pseudo_label(model, unlabelled_data, nb_data, threshold)
    
    
    preds = model.predict(unlabelled_data[0])
    log_pred = -np.log(preds)
    entropy = np.sum(preds*log_pred, axis=1)
    # do entropy
    index = np.argsort(entropy)[::-1]
    
    index_query = index[:nb_data]
    index_unlabelled = index[nb_data:]

    if len(labelled_data[0])==0:
        new_data = unlabelled_data[0][index_query]
        new_labels = unlabelled_data[1][index_query]
    else:
        new_data = np.concatenate([labelled_data[0], unlabelled_data[0][index_query]], axis=0)
        new_labels = np.concatenate([labelled_data[1], unlabelled_data[1][index_query]], axis=0)
    return (new_data, new_labels), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])
           
           
def pseudo_label(model, unlabelled_data, nb_data, threshold):
    # do not consider the real labels
    preds = model.predict(unlabelled_data[0])
    log_pred = -np.log(preds)
    entropy = np.sum(preds*log_pred, axis=1)
    # do entropy
    index = np.argsort(entropy)
    
    delta_index = np.argmin( (entropy[index] < threshold))
    if delta_index==0:
        if entropy[index][0]<threshold:
            return unlabelled_data, ([],[])
        return ([], []), \
               unlabelled_data
    else:
        print('pseudo labelling...')
        delta_index-=1
        index_query = index[:delta_index]
        labels = kutils.to_categorical(np.argmax(preds[index_query], axis=1), num_classes=10)
        index_unlabelled = index[delta_index:]
        
        return (unlabelled_data[0][index_query], labels), \
               (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])
 

def egl_selection(model, unlabelled_data, nb_data):
    
    num_classes = model.get_output_shape_at(0)[-1]
    def get_gradient(model):
        input_shape = model.get_input_shape_at(0)
        output_shape = model.get_output_shape_at(0)
        x = K.placeholder(input_shape)
        y = K.placeholder(output_shape)
        y_pred = model.call(x)
        loss = K.mean(keras.losses.categorical_crossentropy(y, y_pred))
        weights = [tensor for tensor in model.trainable_weights]
        optimizer = model.optimizer
        gradient = optimizer.get_gradients(loss, weights)
    
        return K.function([K.learning_phase(), x, y], gradient)

    f_grad = get_gradient(model)
    
    def compute_egl(image):    
        # test
        grad = []
        
        for k in range(num_classes):
            y_label = np.zeros((1, num_classes))
            y_label[0,k] = 1
            grad_k = f_grad([0, image, y_label])
            grad_k = np.concatenate([np.array(grad_w).flatten() for grad_w in grad_k])
            grad.append(grad_k)
            
        grad = np.mean(grad, axis=0)
        return np.linalg.norm(grad)

    n = min(300, len(unlabelled_data[0]))
    subset_index = np.random.permutation(len(unlabelled_data[0]))
    subset = unlabelled_data[0][subset_index[:n]]
    scores = [compute_egl(subset[i:i+1]) for i in range(len(subset))]
    index = np.argsort(scores)[::-1]
    index_query = subset_index[index[:nb_data]]
    index_unlabelled = np.concatenate( (subset_index[index[nb_data:]], subset_index[n:]))

    return (unlabelled_data[0][index_query], unlabelled_data[1][index_query]), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])


def qbc_selection(model, unlabelled_data, nb_data):
          
    n = min(300, len(unlabelled_data[0]))
    subset_index = np.random.permutation(len(unlabelled_data[0]))
    subset = unlabelled_data[0][subset_index[:n]]
    qbdc_index = np.argsort(disagreement_coeff(subset, model, f=build_model_MNIST))[::-1] # high disagreement
    index_query = subset_index[qbdc_index[:nb_data]]
    index_unlabelled = np.concatenate( (subset_index[qbdc_index[nb_data:]], qbdc_index[n:]))

    return (unlabelled_data[0][index_query], unlabelled_data[1][index_query]), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled]) 


def adversarial_selection(model, unlabelled_data, nb_data):
    active = Adversarial_DeepFool(model=model, n_channels=1,
                                  img_nrows=28, img_ncols=28)
    
    # select a subset of size 10*nb_data
    n = min(300, len(unlabelled_data[0]))
    subset_index = np.random.permutation(len(unlabelled_data[0]))
    subset = unlabelled_data[0][subset_index[:n]]
    adversarial = active.generate(subset)
    index_query = subset_index[adversarial[:nb_data]]
    index_unlabelled = np.concatenate( (subset_index[adversarial[nb_data:]], subset_index[n:]))

    return (unlabelled_data[0][index_query], unlabelled_data[1][index_query]), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])
    

def active_learning(num_sample=32, percentage=0.3, 
                    active_method='adversarial', nb_exp=0, repo='test', filename='test.csv'):
    
    batch_size = num_sample
    labelled_data, unlabelled_data, test_data = build_data(num_sample=num_sample)
    percentage_data = 1.*len(labelled_data[0])/len(unlabelled_data[0])
    # load data
    N_pool = len(labelled_data)+ len(unlabelled_data[0])
    nb_query=10
    n_start = len(labelled_data[0])
    threshold = 0.3
    d_r = 0.0033
    i=0
    while( percentage_data<=percentage):

        #print(i)
        i+=1
        model = active_training(labelled_data, batch_size=batch_size)
        query, unlabelled_data = active_selection(model=model, 
                                                  unlabelled_data=unlabelled_data, 
                                                  active_method=active_method, 
                                                  nb_data=nb_query,
                                                  threshold=threshold)
        
        evaluate(model, percentage_data, test_data, nb_exp, repo, filename)
        
        # add query to the labelled set
        labelled_data_0 = np.concatenate((labelled_data[0], query[0]), axis=0)
        labelled_data_1 = np.concatenate((labelled_data[1], query[1]), axis=0)
        labelled_data = (labelled_data_0, labelled_data_1)
        #update percentage_data
        #percentage_data = 1.*len(labelled_data[0])/len(unlabelled_data[0])
        percentage_data = (n_start+ i*nb_query)/N_pool
        threshold = threshold - d_r
        

        
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Active Learning on MNIST')

    parser.add_argument('--active_method', type=str, default='qbc', help='active learning selection')
    parser.add_argument('--id_experiment', type=int, default=0, help='id number of experiment')
    parser.add_argument('--repo', type=str, default='.', help='repository for log')
    parser.add_argument('--filename', type=str, default='mnist_qbdc', help='csv filename')
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
