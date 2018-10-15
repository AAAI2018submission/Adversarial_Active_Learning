# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:39:37 2017

@author: mducoffe

datasets preprocessing
"""

import numpy as np
import keras.utils.np_utils as kutils
from keras.datasets import mnist, cifar10
import scipy.misc as misc
import keras.backend as K
import os
from contextlib import closing
import h5py
import pickle as pkl

SVHN_PATH='./svhn'
BAG_SHOE_PATH='./dataset'
CIFAR_PATH='./dataset'
QUICK_DRAW='./dataset'

def build_quick_draw(num_sample):
    filenames=['dolphin','cat', 'face','angel']
    x = []
    for filename in filenames:
        data = np.load(os.path.join(QUICK_DRAW, filename+'.npy'))
        x.append(data)
    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]
    for x_data, i in zip(x, range(len(x))):
        n_data = len(x_data)
        n_train_data = (int)(0.8*n_data)
        x_train.append(x_data[:n_train_data])
        x_test.append(x_data[n_train_data:])
        y_train.append([i]*n_train_data)
        y_test.append([i]*(n_data - n_train_data))
    x_train = np.concatenate(x_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    
    # reshape
    x_train = x_train.reshape((x_train.shape[0], 1, 28,28))
    x_test = x_test.reshape((x_test.shape[0], 1, 28,28))
    
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
    
    if num_sample<0:
        x_L=x_train; y_L=y_train
        x_U=[]; y_U=[]
    else:
        x_L = x_train[:num_sample]; y_L = y_train[:num_sample]
        x_U = x_train[num_sample:]; y_U = y_train[num_sample:]

    return (x_L, y_L), (x_U, y_U), (x_test, y_test)
    
    
    

def build_bag_shoes(num_sample):
    
    filename_shoes='shoes_64.hdf5'
    filename_bag='handbag_64.hdf5'
    with closing(h5py.File(os.path.join(BAG_SHOE_PATH, filename_shoes), 'r')) as g_shoe:
        key = g_shoe.keys()[0]
        x_shoes = g_shoe[key][:].transpose((0,3,1,2))
        x_train_shoes = x_shoes[:-2000]
        x_test_shoes = x_shoes[-2000:]
    with closing(h5py.File(os.path.join(BAG_SHOE_PATH, filename_bag), 'r')) as g_bag:
        key = g_bag.keys()[0]
        x_bag = g_bag[key][:].transpose((0,3,1,2))
        x_train_bag = x_bag[:-2000]
        x_test_bag = x_bag[-2000:]
        

    x_train = np.concatenate([x_train_shoes, x_train_bag], axis=0)
    y_train = np.array([0]*len(x_train_shoes)+[1]*len(x_train_bag))
    
    x_test = np.concatenate([x_test_shoes, x_test_bag], axis=0)
    y_test = np.array([0]*len(x_test_shoes)+[1]*len(x_test_bag))
    
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


    if num_sample<0:
        x_L=x_train; y_L= y_train
        x_U=[]; y_U=[]
    else:    
        x_L = x_train[:num_sample]; y_L = y_train[:num_sample]
        x_U = x_train[num_sample:]; y_U = y_train[num_sample:]
    return (x_L, y_L), (x_U[:20000], y_U[:20000]), (x_test, y_test)
    

def build_mnist(num_sample):
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_dim_ordering() == "th":
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
        #input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        
        # we force the data to be channel first
        x_train = x_train.transpose((0,3,1,2))
        x_test = x_test.transpose((0,3,1,2))
    """
    # for VGG we need to resize the data
    if img_rows!=28 or img_cols!=28:
        # we need to resize the image (apply bilinear interpolation)
        x_train_ = [ misc.imresize(x_train[i,0,:,:], (img_rows, img_cols))[None,:,:] for i in range(x_train.shape[0])]
        x_test_ = [ misc.imresize(x_test[i,0,:,:], (img_rows, img_cols))[None,:,:] for i in range(x_test.shape[0])]
        x_train = np.array(x_train_)
        x_test = np.array(x_test_)
    """
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    N = len(x_train)
    index = np.random.permutation(N)
    x_train = x_train[index]
    y_train = kutils.to_categorical(y_train[index])
    

    if num_sample<0:
        x_L=x_train; y_L=y_train
        x_U=[]; y_U=[]
    else:
        x_L = x_train[:num_sample]; y_L = y_train[:num_sample]
        x_U = x_train[num_sample:]; y_U = y_train[num_sample:]
    
    return (x_L, y_L), (x_U, y_U), (x_test, kutils.to_categorical(y_test))
    
    
def build_svhn(num_sample):
    import scipy.io as io
    
    dico_train = io.loadmat(os.path.join(SVHN_PATH,'train_32x32.mat'))
    x_train = dico_train['X']
    x_train = x_train.transpose((3, 0,1, 2))
    y_train = dico_train['y'] -1
    
    dico_test = io.loadmat(os.path.join(SVHN_PATH,'test_32x32.mat'))
    x_test = dico_test['X']
    x_test = x_test.transpose((3,0,1, 2))
    y_test = dico_test['y'] -1

    
    x_train = x_train.astype('float32')
    x_train /= 255.0
    x_test = x_test.astype('float32')
    x_test /= 255.0
    
    x_train = x_train.transpose((0,3,1,2))
    x_test = x_test.transpose((0,3,1,2))
    """
    def resize(img, img_rows, img_cols):
        new_img = np.zeros((3,img_rows, img_cols))
        new_img[0] = misc.imresize(img[0], (img_rows, img_cols))
        new_img[1] = misc.imresize(img[1], (img_rows, img_cols))
        new_img[2] = misc.imresize(img[2], (img_rows, img_cols))
        
        return new_img
    """
    if img_rows!=32 or img_cols!=32:
        # we need to resize the image (apply bilinear interpolation)
        x_train_ = [ resize(x_train[i], img_rows, img_cols) for i in range(x_train.shape[0])]
        x_test_ = [ resize(x_train[i], img_rows, img_cols) for i in range(x_test.shape[0])]
        x_train = np.array(x_train_)
        x_test = np.array(x_test_)
    
    y_train = kutils.to_categorical(y_train)
    y_test = kutils.to_categorical(y_test)
    N = len(x_train)
    index = np.random.permutation(N)
    x_train = x_train[index]
    y_train = y_train[index]
    
    x_L = x_train[:num_sample]; y_L = y_train[:num_sample]
    x_U = x_train[num_sample:]; y_U = y_train[num_sample:]
    
    return (x_L, y_L), (x_U, y_U), (x_test, y_test)

    
def build_cifar(num_sample):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    def resize(img, img_rows, img_cols):
        new_img = np.zeros((3,img_rows, img_cols))
        new_img[0] = misc.imresize(img[0], (img_rows, img_cols))
        new_img[1] = misc.imresize(img[1], (img_rows, img_cols))
        new_img[2] = misc.imresize(img[2], (img_rows, img_cols))
        
        return new_img
    """
    if img_rows!=32 or img_cols!=32:
        # we need to resize the image (apply bilinear interpolation)
        x_train_ = [ resize(x_train[i], img_rows, img_cols) for i in range(x_train.shape[0])]
        x_test_ = [ resize(x_train[i], img_rows, img_cols) for i in range(x_test.shape[0])]
        x_train = np.array(x_train_)
        x_test = np.array(x_test_)
    """
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
    
    x_L = x_train[:num_sample]; y_L = y_train[:num_sample]
    x_U = x_train[num_sample:]; y_U = y_train[num_sample:]
    """    
    x_L = x_L.transpose((0,3,1,2))
    x_U = x_U.transpose((0,3,1,2))
    x_test = x_test.transpose((0,3,1,2))
    """
    return (x_L, y_L), (x_U, y_U), (x_test, y_test)
    
def build_cifar_sift(num_sample,nb_sift=16):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    def resize(img, img_rows, img_cols):
        new_img = np.zeros((3,img_rows, img_cols))
        new_img[0] = misc.imresize(img[0], (img_rows, img_cols))
        new_img[1] = misc.imresize(img[1], (img_rows, img_cols))
        new_img[2] = misc.imresize(img[2], (img_rows, img_cols))
        
        return new_img

    if not(os.path.isfile(os.path.join(CIFAR_PATH, 'cifar_sift.pkl'))):
        x_train_sift=[cv2.cvtColor(img.transpose((1,2,0)),cv2.COLOR_BGR2GRAY) for img in x_train]
        x_test_sift=[cv2.cvtColor(img.transpose((1,2,0)),cv2.COLOR_BGR2GRAY) for img in x_test]
        
        sift = cv2.xfeatures2d.SIFT_create()
        train_sift=[]
        test_sift=[]
        train_sift_y=[]
        test_sift_y=[]
        for gray, label in zip(x_train_sift, y_train):
            kp,des = sift.detectAndCompute(gray,None)
            if len(kp)<nb_sift:
                continue
            else:
                train_sift.append(des[:nb_sift])  
                train_sift_y.append(label)
        for gray,label in zip(x_test_sift, y_test):
            kp,des = sift.detectAndCompute(gray,None)
            if len(kp)<nb_sift:
                continue
            else:
                test_sift.append(des[:nb_sift])
                test_sift_y.append(label)
        
        
        
        x_train = np.array(train_sift)[:,:,:,None]
        x_test = np.array(test_sift)[:,:,:,None]
        y_train = np.array(train_sift_y)
        y_test = np.array(test_sift_y)
        
        with closing(open(os.path.join(CIFAR_PATH, 'cifar_sift.pkl'), 'wb')) as f:
            pkl.dump(((x_train, y_train), (x_test, y_test)), f, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        with closing(open(os.path.join(CIFAR_PATH, 'cifar_sift.pkl'), 'rb')) as f:
            (x_train, y_train), (x_test, y_test)=pkl.load(f)

    x_train = x_train.transpose((0,3,1,2))
    x_test = x_test.transpose((0,3,1,2))
    y_train = kutils.to_categorical(y_train)
    y_test = kutils.to_categorical(y_test)
    N = len(x_train)
    index = np.random.permutation(N)
    x_train = x_train[index]
    y_train = y_train[index]
    
    x_L = x_train[:num_sample]; y_L = y_train[:num_sample]
    x_U = x_train[num_sample:]; y_U = y_train[num_sample:]
    """    
    x_L = x_L.transpose((0,3,1,2))
    x_U = x_U.transpose((0,3,1,2))
    x_test = x_test.transpose((0,3,1,2))
    """
    return (x_L, y_L), (x_U, y_U), (x_test, y_test)

def build_data_func(dataset_name, num_sample):
    dataset_name = dataset_name.lower()
    
    assert (dataset_name in ['mnist', 'svhn', 'cifar', 'bag_shoes', 'quick_draw']), 'unknown dataset {}'.format(dataset_name)
    labelled = None; unlabelled=None; test=None;
    if dataset_name=='mnist':
        labelled, unlabelled, test = build_mnist(num_sample)
    
    if dataset_name=='svhn':
        # TO DO
        labelled, unlabelled, test = build_svhn(num_sample)
    
    if dataset_name=='cifar':
        # TO DO
        labelled, unlabelled, test = build_cifar(num_sample)
        
    if dataset_name=='quick_draw':
        # TO DO
        labelled, unlabelled, test = build_quick_draw(num_sample)
    
    if dataset_name=='bag_shoes':
        labelled, unlabelled, test = build_bag_shoes(num_sample)    
    return labelled, unlabelled, test
    
def getSize(dataset_name):
    dataset_name = dataset_name.lower()
    assert (dataset_name in ['mnist', 'svhn', 'cifar', 'bag_shoes', 'quick_draw']), 'unknown dataset {}'.format(dataset_name)
    
    if dataset_name=='mnist':
        return (1,28,28, 10)
    
    if dataset_name=='svhn':
        return (3,32,32,10)
    
    if dataset_name=='cifar':
        return (3,32,32,10)
        
    if dataset_name=='bag_shoes':
        return (3,64,64,2)
    
    if dataset_name=='quick_draw':
        return (1,28,28, 4)
        
    return None

if __name__=="__main__":
    build_data_func('bag_shoes', 100)
