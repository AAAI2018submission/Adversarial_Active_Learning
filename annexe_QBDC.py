# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:45:03 2017

@author: mducoffe

Query By Dropout Committee
"""
import numpy as np
import keras.backend as K

def dropout_model(model_original, model_committee, dropout_rate=0.5):
    params_original = model_original.trainable_weights
    params_committee = model_committee.trainable_weights
    for param_0, param_1 in zip(params_original, params_committee):
        if param_0.ndim==1:
            continue
        if param_0.ndim==2:
            # TO DO
            param_0_val = param_0.get_value()
            shape = param_0_val.shape
            dropout_mask = np.array([np.random.binomial(1, 0.5) for i in range(np.prod(shape))]).reshape(shape)
            param_1_val = param_0_val*dropout_mask
            param_1.set_value(param_1_val.astype(np.float32))
        if param_0.ndim==4:
            param_0_val = param_0.get_value()
            shape = param_0_val.shape
            dropout_mask = np.ones(shape, dtype='float32')
            if K.image_dim_ordering()=='th':
                for i in range(shape[-1]):
                    if np.random.binomial(1, 0.5)==0:
                        dropout_mask[:,:,:,i]=0
            
            param_1_val = param_0_val*dropout_mask
            param_1.set_value(param_1_val.astype(np.float32))
            
def init_committee(model_original, nb_committee=5, f=None):
    
    #f = locals()[build_name]
    committee = [f() for i in range(nb_committee)]
    for member in committee:
        dropout_model(model_original, member)
        
    return committee
    
        
def disagreement_coeff(unlabelled_data, model_original, nb_committee=5, f=None):
    committee = init_committee(model_original, nb_committee, f)
    committee = [model_original] + committee
    N_data = len(unlabelled_data)
    # majority vote for the predicted label
    predictions = np.array([ member.predict(unlabelled_data) for member in committee]) # (nb_committee, N_data, nb_class)
    labels = np.argmax(np.mean(predictions, axis=0), axis=1) #(N_data, 1)
    
    prob_max = np.max(predictions, axis=2) #(nb_committee, N_data)
    scores = np.zeros((N_data,))
    for i in range(N_data):
        prob_label = predictions[:,i, labels[i]]
        scores[i] = np.mean(prob_max[:,i] - prob_label)

    return scores
        

                    

