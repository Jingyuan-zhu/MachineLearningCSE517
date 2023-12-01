"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    # valid_alphas = np.nonzero((alphas > 0) & (alphas < C))[0]
    # alphas = alphas[valid_alphas]


    # K = K[:,valid_alphas][valid_alphas].reshape(-1,alphas.shape[0])

    # yTr = yTr[valid_alphas]

    # w = np.matmul(K,alphas * yTr)

    # bias = np.mean(yTr - w)
    y_idx = np.argmax(np.abs(alphas*(C - alphas))) #y_idx = np.argmin(np.abs(alphas - C*0.5))

    bias = yTr[y_idx] - np.multiply(yTr, alphas).T.dot(K[y_idx, :]) 


    return bias 
    
