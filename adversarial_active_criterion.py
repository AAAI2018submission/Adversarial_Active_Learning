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
        """
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
        """
        
        img_shape = (1,n_channels, img_nrows, img_ncols)
        self.img_shape=img_shape
        adversarial_image = K.placeholder((1, n_channels, img_nrows, img_ncols))
        adversarial_target = K.placeholder((1, nb_class))
        adv_noise = K.placeholder((1, n_channels, img_nrows, img_ncols))
            
        self.model = model
        
        def get_weights():
            layers = self.model.layers
            norm_weights = 1
            for layer in layers:
                if hasattr(layer, 'kernel'):
                    w = np.linalg.norm(layer.kernel.get_value().flatten())
                    norm_weights*=w
            return norm_weights
        
        self.norm_weights = get_weights()
        
        
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
    
    def prediction(self, image):
        return self.output_([0, image])
        
    def generate_sample(self, true_image):
        raise NotImplementedError()

class Adversarial_Szegedy(Adversarial_example):
    
    def __init__(self,**kwargs):
        super(Adversarial_Szegedy, self).__init__(**kwargs)
        
        loss_classif=K.mean(K.categorical_crossentropy(self.model(self.adv_noise), self.adversarial_target)) +\
                        0.001*K.sum(K.abs(self.adv_noise - self.adversarial_image))

        grad_adversarial = K.gradients(loss_classif, self.adv_noise)

        f_loss = K.function([K.learning_phase(), self.adv_noise, self.adversarial_image, self.adversarial_target], loss_classif)
        f_grad = K.function([K.learning_phase(), self.adv_noise, self.adversarial_image, self.adversarial_target], grad_adversarial)
        
        def eval_loss(adv_label, true_image):
            
            def function(noise):
                r = noise.astype('float32')
                r = r.reshape(self.img_shape)
                x = true_image.astype('float32')
                x = x.reshape(self.img_shape)
                y = np.array([0.]*(self.nb_class))
                # pick random choice
                y[adv_label] = 1.
                y = y[None,:]
                return f_loss([0, r, x, y]).astype('float64')
            return function
        
        self.eval_loss = eval_loss
        
        def eval_grad(adv_label, true_image):
            
            def function(noise):
                r = noise.astype('float32')
                r = r.reshape(self.img_shape)
                x = true_image.astype('float32')
                x = x.reshape(self.img_shape)
                y = np.array([0.]*(self.nb_class))
                # pick random choice
                y[adv_label] = 1.
                y = y[None,:]
                return f_grad([0, r, x, y]).flatten().astype('float64')
            return function
        
        self.eval_grad = eval_grad
        prediction = self.model.call(self.adv_noise)

        
    def generate_sample(self, x, adv_label):

        
        image_adv = np.copy(x.flatten())
        for i in range(100):
            image_adv = image_adv.flatten()
            eval_loss = self.eval_loss(adv_label, image_adv)
            eval_grad = self.eval_grad(adv_label, image_adv)
            noise = np.copy(x).flatten()
            results = scipy.optimize.fmin_l_bfgs_b(eval_loss, noise,
                                                   fprime=eval_grad, 
                                                   maxfun=100)
            noise = results[0].astype('float32')
            image_adv = noise.reshape((self.img_shape[0], self.img_shape[1], self.img_shape[2], self.img_shape[3]))
            label_pred = self.predict(image_adv)
            if label_pred == adv_label:
                return image_adv, True
        #print('FAIL')
        return x, False

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
        
    
    def generate(self, data, option='euclidian'):
        #perturbations=[self.generate_sample(data[i:i+1]) for i in range(len(data))]

        perturbations = []
        adv_attacks = []
        for i in range(len(data)):
            r_i, x_i = self.generate_sample(data[i:i+1], option=option)
            perturbations.append(r_i)
            adv_attacks.append(x_i[0])
            
        """
        # compute also the second bar
        uncertainty = []
        for i in range(len(data)):
            uncertainty.append(self.lower_bound_sample(data[i:i+1]))
        """
        
        index_perturbation = np.argsort(perturbations)
        tmp = np.array(adv_attacks)
        return index_perturbation, tmp[index_perturbation]
        """
        uncertainty = np.array(uncertainty)/self.norm_weights
        index_perturbation = np.argsort(perturbations)
        
        perturbations = perturbations[index_perturbation]
        uncertainty = uncertainty[index_perturbation]
        
        N = len(data)
        sorted_index = np.arange(N)
        
        sorted_index = self.priv_sort_interval(perturbations, uncertainty, sorted_index)
        import pdb; pdb.set_trace()
        index_perturbation = index_perturbation[sorted_index]
        #return np.argsort(perturbations)
        return index_perturbation
        """
    
    def priv_sort_interval(self,array_a, array_b, sorted_index):
        
        # array_a : upper bound
        # array_b : lower bound
        N = len(array_a)
        for i in range(N-1):
            index_0 = sorted_index[i]
            index_1 = sorted_index[i+1]
            if array_a[index_0]<=array_b[index_1]:
                continue
            # array_a[i] > array_b[i+1]
            if array_b[index_1]>=array_b[index_0]:
                continue
            
            if array_b[index_0]> array_b[index_1]:
                proba = (array_b[index_0] - array_b[index_1])/(array_a[index_0] - array_b[index_1])
                if proba >=0.5:
                    
                    sorted_index[i]=index_1
                    sorted_index[i+1]=index_0
                    return self.priv_sort_interval(array_a, array_b, sorted_index)
                else:
                    continue
        return sorted_index
            
    
    def lower_bound_sample(self, true_image):
        true_label = self.predict(true_image)
        f_x = self.model.predict(true_image).flatten()
        
        score = np.inf
        index=-1
        for i in range(self.nb_class):
            if i==true_label:
                continue
            vector = np.zeros((self.nb_class))
            vector[true_label]=np.sqrt(2)
            vector[i]=-np.sqrt(2)
            score_i = np.dot(vector, f_x)
            
            if score_i <score:
                score=score_i
                index=i

        return score
         
    def generate_sample(self, true_image, option='euclidian'):
        assert option in ['euclidian', 'inf'], ('unknown option %s',option)
        if option == 'euclidian':
            return self.generate_sample_euclidian(true_image)
        else:
            return self.generate_sample_infinity(true_image)
    
    def generate_sample_infinity(self, true_image):

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
            result = [f_k/(sum(np.abs(w_k))) for f_k, w_k in zip(f_labels, w_labels)]
            label_adv = np.argmin(result)
            
            r_i = (f_labels[label_adv]/(np.sum(np.abs(w_labels[label_adv]))) )*np.sign(w_labels[label_adv])
            #print(self.predict(x_i), f_labels[label_adv], np.mean(x_i), np.mean(r_i))
            if np.max(np.isnan(r_i))==True:
                return 0, true_image
            x_i += r_i.reshape(true_image.shape)
            #x_i = np.clip(x_i, self.mean - self.std, self.mean+self.std)
            i+=1
            
            
        adv_image = x_i
        adv_label = self.predict(adv_image)
        if adv_label == true_label:
            return np.inf, x_i
        else:
            perturbation = (x_i - true_image).flatten()
            #return np.linalg.norm(perturbation)
            return np.max(np.abs(perturbation)), x_i
        
    
    
    def generate_sample_euclidian(self, true_image):
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
            result = [f_k/(np.linalg.norm(w_k)) for f_k, w_k in zip(f_labels, w_labels)]
            label_adv = np.argmin(result)
            
            r_i = (f_labels[label_adv]/(np.linalg.norm(w_labels[label_adv])**2) )*w_labels[label_adv]
            #print(self.predict(x_i), f_labels[label_adv], np.mean(x_i), np.mean(r_i))
            if np.max(np.isnan(r_i))==True:
                return np.inf, true_image
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

