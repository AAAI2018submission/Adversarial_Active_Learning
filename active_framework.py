# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:03:13 2017

@author: mducoffe
"""

import sys

import numpy as np
import sklearn.metrics as metrics
import argparse
import keras

from keras import backend as K
#from snapshot import SnapshotCallbackBuilder
import csv
from contextlib import closing
import os
from build_model import build_model_func
from build_data import build_data_func, getSize
from adversarial_active_criterion import Adversarial_DeepFool
from bayesian_cnn import bald
import keras.utils.np_utils as kutils
import pickle
import gc
from keras.preprocessing.image import ImageDataGenerator

#%%
import resource
from keras.callbacks import Callback
class MemoryCallback(Callback):
    def on_epoch_end(self, epoch, log={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)



#%%
def active_training(labelled_data, network_name, img_size,
                    batch_size=64, epochs=100, repeat=5):
    
    x_L, y_L = labelled_data 
    
    # split into train and validation
    
    N = len(y_L)
    n_train = (int) (N*0.8)

    batch_train = min(batch_size, len(x_L))
    steps_per_epoch = int(n_train/batch_train)+1
    best_model = None
    best_loss = np.inf
    for i in range(repeat):
        # shuffle data and split train and val
        index = np.random.permutation(N)
        x_train , y_train = (x_L[index[:n_train]], y_L[index[:n_train]])
        x_val , y_val = (x_L[index[n_train:]], y_L[index[n_train:]])
        
        generator_train = ImageDataGenerator()
        generator_train.fit(x_train, seed=0, augment=True)
        tmp = generator_train.flow(x_train, y_train, batch_size=batch_size)
        model = build_model_func(network_name, img_size)
        earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        hist = model.fit_generator(tmp, steps_per_epoch, epochs=epochs,
                                   verbose=0,
                                   callbacks=[earlyStopping],
                                   validation_data=(x_val, y_val))
                                   
        loss, acc = model.evaluate(x_val, y_val, verbose=0)
        if loss < best_loss:
            best_loss = loss;
            best_model = model

    del model
    del hist
    del loss
    del acc
    i=gc.collect()
    while(i!=0):
        i=gc.collect()
    return best_model

#%%
def evaluate(model, percentage, test_data, nb_exp, repo, filename):
    x_test, y_test = test_data
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    
    with closing(open(os.path.join(repo, filename), 'a')) as csvfile:
        # TO DO
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([str(nb_exp), str(percentage), str(acc)])
         
    #return query, unlabelled_pool

#%%
def get_weights(model):
    layers = model.layers
    weights=[]
    for layer in layers:
        if layer.trainable_weights:
            weights_layer = layer.trainable_weights
            weights+=[elem.get_value() for elem in weights_layer]
    return weights
    
def load_weights(model, weights):
    layers = model.layers
    index=0
    for layer in layers:
        if layer.trainable_weights:
            weights_layer = layer.trainable_weights
            for elem in weights_layer:
                elem.set_value(weights[index])
                index+=1
    return model
                
                
def loading(repo, filename, num_sample, network_name, data_name):
    # check if file exists
    img_size = getSize(data_name) # TO DO
    model=build_model_func(network_name, img_size)
    filename = filename.split('.pkl')
    f_weights = filename[0]+'_weights.pkl'
    f_l_data = filename[0]+'_labelled.pkl'
    f_u_data = filename[0]+'_unlabelled.pkl'
    f_t_data = filename[0]+'_test.pkl'
    if (os.path.isfile(os.path.join(repo, f_weights)) and \
        os.path.isfile(os.path.join(repo, f_l_data)) and \
        os.path.isfile(os.path.join(repo, f_u_data)) and \
        os.path.isfile(os.path.join(repo, f_t_data))):
        
        
        
        with closing(open(os.path.join(repo, f_weights), 'rb')) as f:
            weights = pickle.load(f)
            model = load_weights(model, weights)
            
        with closing(open(os.path.join(repo, f_l_data), 'rb')) as f:
            labelled_data = pickle.load(f)   
            
        with closing(open(os.path.join(repo, f_u_data), 'rb')) as f:
            unlabelled_data = pickle.load(f) 
            
        with closing(open(os.path.join(repo, f_t_data), 'rb')) as f:
            test_data = pickle.load(f)
    else:
        # TO DO !!!
        print('no previous savings, starting from scratch')
        labelled_data, unlabelled_data, test_data = build_data_func(data_name, num_sample=num_sample)
    
    return model, labelled_data, unlabelled_data, test_data

def saving(model, labelled_data, unlabelled_data, test_data, repo, filename):
    weights = get_weights(model)
    #data = (weights, labelled_data, unlabelled_data, test_data)
    
    filename = filename.split('.pkl')
    f_weights = filename[0]+'_weights.pkl'
    f_l_data = filename[0]+'_labelled.pkl'
    f_u_data = filename[0]+'_unlabelled.pkl'
    f_t_data = filename[0]+'_test.pkl'
    
    with closing(open(os.path.join(repo, f_weights), 'wb')) as f:
        pickle.dump(weights, f)
    with closing(open(os.path.join(repo, f_l_data), 'wb')) as f:
        pickle.dump(labelled_data, f)
    with closing(open(os.path.join(repo, f_u_data), 'wb')) as f:
        pickle.dump(unlabelled_data, f)
    with closing(open(os.path.join(repo, f_t_data), 'wb')) as f:
        pickle.dump(test_data, f)

#%%

def active_selection(model, unlabelled_data, nb_data, active_method, repo, tmp_adv):
    assert active_method in ['uncertainty', 'egl', 'random', 'aaq', 'saaq', 'ceal', 'bayesian'], ('Unknown active criterion %s', active_method)
    if active_method=='uncertainty':
        query, unlabelled_data = uncertainty_selection(model, unlabelled_data, nb_data)
    if active_method=='random':
        query, unlabelled_data = random_selection(unlabelled_data, nb_data)
    if active_method=='egl':
        query, unlabelled_data = egl_selection(model, unlabelled_data, nb_data)
    if active_method=='aaq':
        tmp_adv=None
        query, unlabelled_data = adversarial_selection(model, unlabelled_data, nb_data, False, repo, tmp_adv)
    if active_method=='saaq':
        tmp_adv=None
        query, unlabelled_data = adversarial_selection(model, unlabelled_data, nb_data, True, repo, tmp_adv)    
    if active_method=='ceal':
        query, unlabelled_data = ceal_selection(model, unlabelled_data, nb_data)
    if active_method=='bayesian':
        query, unlabelled_data = bald_selection(model, unlabelled_data, nb_data)
        
    return query, unlabelled_data
    
def random_selection(unlabelled_data, nb_data):
    index = np.random.permutation(len(unlabelled_data[0]))
    index_query = index[:nb_data]
    index_unlabelled = index[nb_data:]
    
    return (unlabelled_data[0][index_query], unlabelled_data[1][index_query]), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])
           
def bald_selection(model, unlabelled_data, nb_data):
    n = min(100, len(unlabelled_data[0]))
    subset_index = np.random.permutation(len(unlabelled_data[0]))
    subset = unlabelled_data[0][subset_index[:n]]
    index = bald(subset, model, 10)
        
    index_query = subset_index[index[:nb_data]]
    index_unlabelled = subset_index[index[nb_data:]]

    new_data = unlabelled_data[0][index_query]
    new_labels = unlabelled_data[1][index_query]

    return (new_data, new_labels), \
           (np.concatenate([unlabelled_data[0][index_unlabelled], unlabelled_data[0][subset_index[n:]]], axis=0), np.concatenate([unlabelled_data[1][index_unlabelled], unlabelled_data[1][subset_index[n:]]], axis=0))

# add CEAL
def uncertainty_selection(model, unlabelled_data, nb_data):

    preds = model.predict(unlabelled_data[0])
    log_pred = -np.log(preds)
    entropy = np.sum(preds*log_pred, axis=1)
    # do entropy
    index = np.argsort(entropy)[::-1]
    
    index_query = index[:nb_data]
    index_unlabelled = index[nb_data:]

    new_data = unlabelled_data[0][index_query]
    new_labels = unlabelled_data[1][index_query]
    """
    else:
        new_data = np.concatenate([labelled_data[0], unlabelled_data[0][index_query]], axis=0)
        new_labels = np.concatenate([labelled_data[1], unlabelled_data[1][index_query]], axis=0)
    """
    return (new_data, new_labels), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])
           

def pseudo_label(model, unlabelled_data, nb_data, threshold):
    # do not consider the real labels
    n = min(300, len(unlabelled_data[0]))
    subset_index = np.random.permutation(len(unlabelled_data[0]))
    subset = unlabelled_data[0][subset_index[:n]]
    
    preds = model.predict(subset)
    log_pred = -np.log(preds)
    entropy = np.sum(preds*log_pred, axis=1)
    # do entropy
    index = np.argsort(entropy)
    
    delta_index = np.argmin( (entropy[index] < threshold))
    if delta_index==0:
        if entropy[index][0]<threshold:
            labelled_data=(unlabelled_data[0][subset_index[:n]], unlabelled_data[1][subset_index[:n]])
            unlabelled_data=(unlabelled_data[0][subset_index[n:]], unlabelled_data[1][subset_index[n:]])
            return labelled_data, unlabelled_data
        return ([],[]), unlabelled_data
        #return unlabelled_data, ([],[])
        #return ([], []), \
        #       unlabelled_data
    else:
        print('pseudo labelling...')
        delta_index-=1
        index_query = index[:delta_index]
        labels = kutils.to_categorical(np.argmax(preds[index_query], axis=1), num_classes=10)
        index_unlabelled = index[delta_index:]
        
        labelled_data=(unlabelled_data[0][index_query], labels)
        unlabelled_data=(np.concatenate([unlabelled_data[0][subset_index[n:]], unlabelled_data[0][index_unlabelled]],axis=0),\
                         np.concatenate([unlabelled_data[1][subset_index[n:]], unlabelled_data[1][index_unlabelled]],axis=0))
        
        return labelled_data, unlabelled_data
        #return (unlabelled_data[0][index_query], labels), \
        #       (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])
               
               
def ceal_selection(model, unlabelled_data, nb_data):
    # consider the lowest entropy for pseudo labelling
    N_data = model.get_output_shape_at(0)[-1]
    threshold=0
    if N_data==10:
        threshold=0.05
    if N_data==4:
        threshold=0.08
    if N_data==2:
        threshold=0.19
    
    threshold=0.002
    labelled_data, unlabelled_data = pseudo_label(model, unlabelled_data, nb_data, threshold)

    preds = model.predict(unlabelled_data[0])
    log_pred = -np.log(preds)
    entropy = np.sum(preds*log_pred, axis=1)

    # do entropy
    index = np.argsort(entropy)[::-1]
    
    index_query = index[:nb_data]
    index_unlabelled = index[nb_data:]

    new_data = unlabelled_data[0][index_query]
    new_labels = unlabelled_data[1][index_query]
    """
    else:
        new_data = np.concatenate([labelled_data[0], unlabelled_data[0][index_query]], axis=0)
        new_labels = np.concatenate([labelled_data[1], unlabelled_data[1][index_query]], axis=0)
    """
    return (new_data, new_labels), \
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
           
def adversarial_selection(model, unlabelled_data, nb_data, add_adv=False, repo='.', filename = None):
    img_size = model.get_input_shape_at(0)
    n_channels, img_nrows, img_ncols = img_size[1:]
    nb_classes = model.get_output_shape_at(0)[-1]
    active = Adversarial_DeepFool(model=model, n_channels=n_channels,
                                  img_nrows=img_nrows, img_ncols=img_ncols, nb_class=nb_classes)
    # select a subset of size 10*nb_data
    n = min(300, len(unlabelled_data[0]))
    subset_index = np.random.permutation(len(unlabelled_data[0]))
    subset = unlabelled_data[0][subset_index[:n]]
    # here consider or not the adv examples for pseudo labelling
    # pick option
    adversarial, attacks = active.generate(subset)
        
    if not(filename is None):
        # save the first adv
        img = unlabelled_data[0][subset_index[adversarial[0]]]
        adv_img = attacks[0]
        #save_adv(repo, filename, img, adv_img)
    index_query = subset_index[adversarial[:nb_data]]
    index_unlabelled = np.concatenate( (subset_index[adversarial[nb_data:]], subset_index[n:]))
    
    if add_adv:
        new_data = np.concatenate([unlabelled_data[0][index_query], attacks[:nb_data]], axis=0)
        new_labels = np.concatenate([unlabelled_data[1][index_query], unlabelled_data[1][index_query]], axis=0)
        
        return (new_data, new_labels), \
              (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])
    else:
        return (unlabelled_data[0][index_query], unlabelled_data[1][index_query]), \
               (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])
               
def save_adv(repo, filename, img, adv_img):
    i = 0
    assert os.path.isdir(repo), ('unknown repository %s', repo)
    while os.path.isfile(os.path.join(repo, filename+'_'+str(i)+'.pkl')):
        i+=1
        
    filename = os.path.join(repo, filename+'_'+str(i)+'.pkl')
    
    with closing(open(filename, 'wb')) as f:
        pickle.dump([img, adv_img], f, protocol =pickle.HIGHEST_PROTOCOL)

#%%
def active_learning(num_sample, data_name, network_name, active_name,
                    nb_exp=0, nb_query=10, repo='test', filename='test.csv'):
    
    # create a model and do a reinit function
    tmp_filename = 'tmp_{}_{}_{}.pkl'.format(data_name, network_name, active_name)
    tmp_adv = None
    if active_name in ['aaq', 'saaq']:
        tmp_adv = 'adv_{}_{}_{}'.format(data_name, network_name, active_name)
    filename = filename+'_{}_{}_{}'.format(data_name, network_name, active_name)
    img_size = getSize(data_name)
    # TO DO filename
    
    model, labelled_data, unlabelled_data, test_data = loading(repo, tmp_filename, num_sample, network_name, data_name)
    batch_size = 32
    percentage_data = len(labelled_data[0])
    N_pool = len(labelled_data[0]) + len(unlabelled_data[0])
    print('START')
    # load data
    i=0
    while( percentage_data<=N_pool):

        i+=1
        model = active_training(labelled_data, network_name, img_size, batch_size=batch_size)
        
        query, unlabelled_data = active_selection(model, unlabelled_data, nb_query, active_name, repo, tmp_adv) # TO DO
        print('SUCCEED')
        evaluate(model, percentage_data, test_data, nb_exp, repo, filename)
        # SAVE
        saving(model, labelled_data, unlabelled_data, test_data, repo, tmp_filename)
        #print('SUCEED')
        #print('step B')
        i=gc.collect()
        while(i!=0):
            i = gc.collect()

        # add query to the labelled set
        labelled_data_0 = np.concatenate((labelled_data[0], query[0]), axis=0)
        labelled_data_1 = np.concatenate((labelled_data[1], query[1]), axis=0)
        labelled_data = (labelled_data_0, labelled_data_1)
        #update percentage_data
        percentage_data +=nb_query
        
#%%
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Active Learning')

    parser.add_argument('--id_experiment', type=int, default=4, help='id number of experiment')
    parser.add_argument('--repo', type=str, default='.', help='repository for log')
    parser.add_argument('--filename', type=str, default='test_0', help='csv filename')
    parser.add_argument('--num_sample', type=int, default=10, help='size of the initial training set')
    parser.add_argument('--data_name', type=str, default='bag_shoes', help='dataset')
    parser.add_argument('--network_name', type=str, default='LeNet5', help='network')
    parser.add_argument('--active', type=str, default='ceal', help='active techniques')
    args = parser.parse_args()
                                                                                                             



                                                                                                                

    nb_exp = args.id_experiment
    repo=args.repo
    filename=args.filename
    if filename.split('.')[-1]=='csv':
        filename=filename.split('.csv')[0]
        
    data_name = args.data_name
    network_name = args.network_name
    active_option = args.active
    num_sample = args.num_sample
    
    active_learning(num_sample=num_sample,
                    data_name=data_name,
                    network_name=network_name,
                    active_name=active_option,
                    nb_exp=nb_exp,
                    repo=repo,
                    filename=filename)


