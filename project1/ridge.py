
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    # YOUR CODE HERE
    loss = lambdaa * np.matmul(w.T,w)
    gradient = 2*lambdaa *w
    for i in range (yTr.shape[1]):
        temp  = np.matmul(w.T,xTr[:,i])-yTr[:,i]
        loss += temp ** 2
        gradient += 2*temp*(xTr[:,i].reshape(-1,1))
    return loss,gradient
