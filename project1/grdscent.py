
import numpy as np
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-09):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent

    w = w0
    step = 0
    difference = np.inf
    gradient = np.inf
    previous_loss = np.inf
    while(np.linalg.norm(gradient) > tolerance and step < maxiter):

        loss,gradient = func(w)
        difference = loss - previous_loss
        previous_loss = loss
        if difference > 0:
            stepsize = 0.5*stepsize
        else:
            stepsize = 1.01*stepsize
        w = w - stepsize * gradient
        step += 1
    # w = w0
    # step = 0
    # difference = np.inf
    # previous_loss = np.inf
    # while(np.linalg.norm(w)>tolerance and step < maxiter):
        
    #     loss,gradient = func(w)
    #     difference = loss - previous_loss
    #     previous_loss = loss
    #     if difference > 0:
    #         stepsize = 0.5*stepsize
    #     else:
    #         stepsize = 1.01*stepsize
    #     w = w - stepsize * gradient
    #     step += 1


    return w
