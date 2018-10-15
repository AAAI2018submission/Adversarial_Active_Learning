# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:30:26 2017

@author: mducoffe

SVD regularization
"""
import keras
import keras.backend as K
import numpy as np
from numpy.linalg import norm
from keras.regularizers import Regularizer
from keras.constraints import Constraint

def get_SVD(Mat):
    #Mat = mat.get_value()
    n = Mat.shape[0]
    m = Mat.shape[1]
    
    v_0 = np.random.ranf(m)
    
    error = 11
    for i in range(1):
        w_1=np.dot(Mat, v_0)
        alpha_1=norm(w_1)
        u_1=w_1/alpha_1
        z_1=np.dot(Mat.T, u_1)
        beta_1=norm(z_1)
        v_1=z_1/beta_1
        
        v_0=v_1
        
        error = norm( np.dot(Mat, v_1)-beta_1*u_1)
    return beta_1, u_1.astype('float32'), v_1.astype('float32')
    # return u_1, v_1
    
def get_SVD_support(mat, U, V):
    Mat = mat.get_value()
    _, u, v = get_SVD(Mat)
    U.set_value(u)
    V.set_value(v)

#%%    
def get_SVD_gpu(Mat):
    n = Mat.shape[0]
    m = Mat.shape[1]
    

#%%    
class SVD_Lipschitz(Regularizer):
    """Regularizer for maximum singular value.
    # Arguments
    """

    def __init__(self, alpha=1.):
        self.alpha=alpha
        self.shape=None
        self.U=None
        self.V=None
        
    def svd(self, x):
        if self.shape is None:
            self.shape=x.shape.eval()
            self.U=K.variable(np.zeros((self.shape[0],)))
            self.V=K.variable(np.zeros((self.shape[1],)))
            
        get_SVD_support(x, self.U, self.V)
        return K.dot(self.U, K.dot(x, self.V))

    def __call__(self, x):
        regularization = 0.
     
        if x.ndim==2:
            regularization=self.svd(x)
        return self.alpha*regularization

    def get_config(self):
        return {}
        
def get_regularizer(reg_name, reg_value):
    if reg_name is None:
        return None
    assert (reg_name in ['l2', 'spectral']), ('unknown regularization method {}'.format(reg_name))
    
    reg_obj=None
    if reg_name=='spectral':
        reg_obj= SVD_Lipschitz(reg_value)
    if reg_name=='jacobian':
        raise NotImplementedError()
    if reg_name=='l2':
        reg_obj=keras.regularizers.l2(0.)
        
    return reg_obj
        
        


        


