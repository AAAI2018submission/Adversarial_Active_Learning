# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:10:27 2017

@author: mducoffe

Bayesian CNN
"""
import keras.backend as K
import numpy as np

def predict_bayesian(model):
    
    f = K.function([K.learning_phase(), model.get_input_at(0)], model.get_output_at(0))
    
    def function(x):
        return f([1, x])
        
    return function
    
def bald(data, model, T):
    f_bayes = predict_bayesian(model)
    samples = np.array([ f_bayes(data) for i in range(T)]) # shape (T, N, c)
    var_A = (1./T)*np.sum(samples, axis=0)
    var_B = np.log(var_A)
    
    var_C = (1./T)*np.sum(samples*np.log(samples), axis=(0,2))
    
    bald_scores = -np.sum(var_A*var_B, axis=1) + var_C
    
    index = np.argsort(bald_scores)[::-1]
    
    return index
    
    
    

