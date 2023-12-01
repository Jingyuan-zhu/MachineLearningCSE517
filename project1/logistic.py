import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):

    # YOUR CODE HERE
    loss = 0
    gradient = 0
    # YOUR CODE HERE
    for i in range(yTr.shape[1]):
        temp = math.exp(-yTr[:,i] * np.matmul(w.T,xTr[:,i]))
        loss += np.log(1+temp)
        gradient -= (((xTr[:,i])*yTr[:,i])/(1+math.exp(yTr[:,i] * np.matmul(w.T,xTr[:,i])))).reshape(-1,1)
    return loss,gradient
