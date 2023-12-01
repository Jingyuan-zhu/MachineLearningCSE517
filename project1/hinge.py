from ast import Lambda
from math import lgamma
from numpy import maximum
import numpy as np

def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regularization constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
    loss = lambdaa * np.matmul(w.T,w)
    gradient = 2*lambdaa *w
    for i in range (yTr.shape[1]):
        temp  = np.matmul(w.T,xTr[:,i])-yTr[:,i]
        loss += temp ** 2
        gradient += 2*temp*(xTr[:,i].reshape(-1,1))


    loss = lambdaa * np.matmul(w.T,w)
    gradient = 2*lambdaa *w
    # YOUR CODE HERE
    for i in range (yTr.shape[1]):
        temp = np.dot(w.T,xTr[:,i])*yTr[:,i]
        loss += max(1-temp,0) 
        if 1-temp > 0:
            gradient += (-yTr[:,i]*xTr[:,i]).reshape(-1,1)
        else: 
            gradient += 0
    return loss,gradient
