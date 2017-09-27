# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.

author : mducoffe

Step 1 : deep fool as an active learning criterion
"""
import numpy as np
import keras.backend as K
import scipy
from contextlib import closing
import pickle as pkl
import os
from keras.models import Model


class Adversarial_example(object):
    
    def __init__(self, model, n_channels=3, img_nrows=32, img_ncols=32, 
                 nb_class=10):

        if K.image_dim_ordering() == 'th':
            img_shape = (1, n_channels, img_nrows, img_ncols)
            adversarial_image = K.placeholder((1, n_channels, img_nrows, img_ncols))
            adversarial_target = K.placeholder((1, nb_class))
            adv_noise = K.placeholder((1, n_channels, img_nrows, img_ncols))
        else:
            img_shape = (1,img_nrows, img_ncols, n_channels)
            adversarial_image = K.placeholder((1, img_nrows, img_ncols, n_channels))
            adversarial_target = K.placeholder((1, nb_class))
            adv_noise = K.placeholder((1, img_nrows, img_ncols, n_channels))
            
        self.model = model
        
        """
        self.model.trainable=False
        for layer in self.model.layers:
            layer.trainable=False
        """
        self.adversarial_image= adversarial_image
        self.adversarial_target = adversarial_target
        self.adv_noise = adv_noise
        self.img_shape = img_shape
        self.nb_class = nb_class
        
        
        prediction = self.model.call(self.adversarial_image)
        self.predict_ = K.function([K.learning_phase(), self.adversarial_image], K.argmax(prediction, axis=1))

        
    def generate(data):
        raise NotImplementedError()
        
    def predict(self,image):
        return self.predict_([0, image])
        
    def generate_sample(self, true_image):
        raise NotImplementedError()



class Adversarial_DeepFool(Adversarial_example):
    
    def __init__(self,  **kwargs):
        super(Adversarial_DeepFool, self).__init__(**kwargs)
        
        # HERE check for the softmax
        
        # the network is evaluated without the softmax
        # you need to retrieve the last layer (Activation('softmax'))
        last_dense = self.model.layers[-2].output
        second_model = Model(self.model.input, last_dense)
        loss_classif = K.mean(second_model.call(self.adversarial_image)[0, K.argmax(self.adversarial_target)])
        grad_adversarial = K.gradients(loss_classif, self.adversarial_image)
        self.f_loss = K.function([K.learning_phase(), self.adversarial_image, self.adversarial_target], loss_classif)
        self.f_grad = K.function([K.learning_phase(), self.adversarial_image, self.adversarial_target], grad_adversarial)
        
        def eval_loss(x,y):
            y_vec = np.zeros((1, self.nb_class))
            y_vec[:,y] +=1
            return self.f_loss([0., x, y_vec])
        
        def eval_grad(x,y):
            y_vec = np.zeros((1, self.nb_class))
            y_vec[:,y] +=1
            return self.f_grad([0., x, y_vec]) 
        
        self.eval_loss = eval_loss
        self.eval_grad = eval_grad
        
    
    def generate(self, data):
        """
        perturbations=[self.generate_sample(data[i:i+1]) for i in range(len(data))]
        """
        
        perturbations = []
        adv_images = []
        for i in range(len(data)):
            r_i, x_i = self.generate_sample(data[i:i+1])
            perturbations.append(r_i)
            adv_images.append(x_i[0])
        
        #return np.argsort(perturbations)
        index_perturbation = np.argsort(perturbations)
        tmp = np.array(adv_images)
        return index_perturbation, tmp[index_perturbation]

    def generate_sample(self, true_image):

        true_label = self.predict(true_image)

        x_i = np.copy(true_image); i=0
        while self.predict(x_i) == true_label and i<10:
            other_labels = range(self.nb_class)
            other_labels.remove(true_label)
            w_labels=[]; f_labels=[]
            for k in other_labels:
                w_k = (self.eval_grad(x_i,k).flatten() - self.eval_grad(x_i, true_label).flatten())
                f_k = np.abs(self.eval_loss(x_i, k).flatten() - self.eval_loss(x_i, true_label).flatten())
                w_labels.append(w_k); f_labels.append(f_k)
            #result = [f_k/(np.linalg.norm(w_k)) for f_k, w_k in zip(f_labels, w_labels)]
            result = [f_k/(sum(np.abs(w_k))) for f_k, w_k in zip(f_labels, w_labels)]
            label_adv = np.argmin(result)
            
            #r_i = (f_labels[label_adv]/(np.linalg.norm(w_labels[label_adv])**2) )*w_labels[label_adv]
            
            r_i = (f_labels[label_adv]/(np.sum(np.abs(w_labels[label_adv]))) )*np.sign(w_labels[label_adv])
            #print(self.predict(x_i), f_labels[label_adv], np.mean(x_i), np.mean(r_i))
            if np.max(np.isnan(r_i))==True:
                return False, true_image, true_image, true_label
            x_i += r_i.reshape(true_image.shape)
            #x_i = np.clip(x_i, self.mean - self.std, self.mean+self.std)
            i+=1
            
            
        adv_image = x_i
        adv_label = self.predict(adv_image)
        if adv_label == true_label:
            return np.inf, x_i
        else:
            perturbation = (x_i - true_image).flatten()
            return np.linalg.norm(perturbation), x_i

