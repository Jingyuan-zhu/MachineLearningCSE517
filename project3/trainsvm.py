"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
C   : regularization constant (in front of loss)
ktype : 'linear', 'rbf', 'polynomial'
P : parameter passed to kernel

Output:
svmclassify : a classifier, svmclassify(xTe), that returns the predictions 1 or -1 on xTe

Trains an SVM classifier with kernel (ktype) and parameters (C, P) on the data set (xTr,yTr)
"""

import numpy as np
from computeK import computeK
from generateQP import generateQP
from recoverBias import recoverBias
from qpsolvers import solve_qp
from createsvmclassifier import createsvmclassifier

def trainsvm(xTr,yTr, C, ktype, P):    
    n = yTr.shape[0]
    K = computeK(ktype, xTr, xTr, P)

    Q, p, G, h, A, b = generateQP(K, yTr, C)

    sol = solve_qp(Q, p.reshape(n,), G, h.reshape(2*n,), A.reshape(n,), b.reshape(1,), solver="cvxopt")
    alphas = np.array(sol.reshape((sol.shape[0], 1)))
    bias = recoverBias(K,yTr,alphas,C)
    svmclassify = createsvmclassifier(xTr, yTr, alphas, bias, ktype, P)
    
    return svmclassify


    
    