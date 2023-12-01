# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:05:03 2019

@author: Jerry Xing
"""
import numpy as np
def preprocess(xTr,xTe):
# function [xTr,xTe,u,m]=preprocess(xTr,xTe);
#
# Preproces the data to make the training features have zero-mean and
# standard-deviation 1
# input:
# xTr - raw training data as d by n_train numpy ndarray 
# xTe - raw test data as d by n_test numpy ndarray
    
# output:
# xTr - pre-processed training data 
# xTe - pre-processed testing data
#
# u,m - any other data should be pre-processed by x-> u*(x-m)
#       where u is d by d ndnumpy array and m is d by 1 numpy ndarray
    
    d, n = np.shape(xTr)
    d_, n_ = np.shape(xTe)  
    ## << Remove 2 lines above and insert your solution here
    # first modify the m and u to calculate the mean and standard deviation of features
    # m should calculate the mean for all features
    m = np.mean(xTr, axis = 1)[:,None] # this creates a 2x1 array
    sigma = np.std(xTr, axis = 1)[:, None] # this creates a 2x1 array of std of each feature
    avg_xTr = np.repeat(m, n, axis = 1)
    avg_xTe = np.repeat(m, n_, axis = 1) #use the transformation to the test data
    # next, calculate u
    u = 1 / np.repeat(sigma, n, axis = 1)
    u = np.diag(np.diag(u))
    print(u.shape, xTr.shape, )

    xTr = u @ (xTr - avg_xTr)
    xTe = u @ (xTe - avg_xTe)

    ## >>
    return xTr, xTe, u, m