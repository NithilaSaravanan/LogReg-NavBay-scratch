# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:55:14 2020

@author: mariliapc
"""

import numpy as np


def sparse_var(a, axis=None):
    """ Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    """
    a_squared = a.copy()
    a_squared = np.power(a_squared.data,2)
    return a_squared.mean(axis) - np.square(a.mean(axis))

def isfloat(val):
    val = val.replace('.', '0', 1)
    val = val.replace('+', '0', 1)
    val = val.replace('-', '0', 1)
    val = val.replace(' ', '0')
    return val.isdigit()

# Logistic sigmoid function
def sigmoid_func(a):    
    return 1 / (1 + np.exp(-a))

# Defining the cost function for Logistic Regression
def LossFunc(Inp, Tar, omg, lam):   
    y = sigmoid_func(np.dot(Inp, omg))
    a = np.multiply((1 - Tar), np.log(1 - y))
    b = np.multiply(Tar, np.log(y))
    return np.mean((-1/len(Inp)) * np.sum(a + b))