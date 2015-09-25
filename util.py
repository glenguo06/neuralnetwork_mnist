# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:03:45 2015

@author: GlennGuo
"""

import scipy as sci
import numpy as np

def MNIST_Scale(x):
    """
    Scale function defined specifically for MNIST
    Reference to publication:
        Ciresan D C, Meier U, Gambardella L M, et al. 
        'Deep Big Simple Neural Nets Excel on Handwritten Digit Recognition'
        Neural Comput.Volume 22, Number 12, December 2010
    
    In:
        x: list/ndarray with pixel density in the range 0 (background) and 255 (foreground)
    Out:
        for each element x_i in x, x_i = x_i/127.5 - 1
    """
    
    return x/127.5 - 1.0

def MNIST_Load(filePath = 'mnist_uint8.mat'):
    """
    Use Scipy.io module to load MNIST mat file
    
    In:
        filePath: give specific path if .mat file is not existed in current directory
        
    Out: 
        [1] train_x OF TYPE float32
        [2] train_y ```
        [3] valid_x ```
        [4] valid_y ```
        [5] test_x ```
        [6] test_y ```
        
    Example :
        train_x, train_y, valid_x, valid_y, test_x, test_y = MNIST_Load()
        
    """
    
    data = sci.io.loadmat(filePath)
    train_x = np.array(data['train_x'][0:50000, :], dtype=np.float32)
    train_y = data['train_y'][0:50000, :]
    valid_x = np.array(data['train_x'][50000:, :], dtype=np.float32)
    valid_y = data['train_y'][50000:, :]
    test_x = np.array(data['test_x'], dtype=np.float32)
    test_y = data['test_y']
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y
    
def Z_score_Scale(x):
    """
    Scale input x with z-score Normalization
    
    This function is created because the limitation of computer memory
    """
    mean_x = np.mean(x, axis = 0)
    var_x = np.var(x, axis = 0)
    
    for i_row in range(len(x)):
        scaled_row = (x[i_row,:] - mean_x)/np.sqrt(var_x)
        x[i_row,:] = scaled_row
        

def MapMinMax(x, ymin = -1.0, ymax = 1.0):
    """
    Map input x into the range(ymin, ymax)
    
    Reference:
        Matlab 6.0( or higer) built in function: 
            x = mapminmax()
    
    Algorithm:
        y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin
    """
    facto = np.tile((np.max(x,axis=1)-np.min(x,axis=1)).reshape(len(x),1), (1, x.shape[1]))
    xmin = np.tile(np.min(x, axis=1).reshape(len(x),1), (1,x.shape[1]))
    
    x = (ymax-ymin)*np.divide((x-xmin),facto) + ymin
    
    return x
