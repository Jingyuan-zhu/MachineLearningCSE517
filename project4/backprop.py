# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:07:53 2019

@author: Jerry Xing
"""
import numpy as np
def backprop(W, aas,zzs, yTr,  trans_func_der):
#% function [gradient] = backprop(W, aas, zzs, yTr,  der_trans_func)
#%
#% INPUT:
#% W weights (list of ndarray)
#% aas output of forward pass (list of ndarray)
#% zzs output of forward pass (list of ndarray)
#% yTr 1xn ndarray (each entry is a label)
#% der_trans_func derivative of transition function to apply for inner layers
#%
#% OUTPUTS:
#% 
#% gradient = the gradient at w as a list of ndarries
#%

    n = np.shape(yTr)[1]
    delta = zzs[0] - yTr
    
    # compute gradient with back-prop
    gradient = [None] * len(W)
    for i in range(len(W)):
        
    	# INSERT CODE HERE:
        # for the update rule, the general form is W(L) <- W(L) - lr * ((Z(L) - y)* g'L(a(L)) @ Z(L-1).T)
        gradient[i] = (1/n) * delta @ zzs[i+1].T
        delta = np.multiply((W[i][:, :-1]).T @ delta , trans_func_der(aas[i + 1]))

    return gradient 


