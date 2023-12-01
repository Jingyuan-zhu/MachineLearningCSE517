"""

INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
C : regularization constant

Output:
Q,p,G,h,A,b as defined in qpsolvers.solve_qp

A call of qpsolvers.solve_qp(Q, p, G, h, A, b) should return the optimal nx1 vector of alphas
of the SVM specified by K, yTr, C. Just make these variables np arrays.

"""
#from tkinter import Y
import numpy as np

def generateQP(K, yTr, C):
    yTr = yTr.astype(np.double)
    n = yTr.shape[0]
    
    # YOUR CODE HERE
    Q = Q = np.matmul(yTr,yTr.T)*K
    p = np.ones((n,1)) * -1
    G_greater = np.diag(np.ones(n)) * -1
    G_less = np.diag(np.ones(n))
    G = np.vstack((G_greater, G_less))
    h = np.hstack((np.zeros(n), np.ones(n) * C)).reshape(-1,1)
    A = yTr.T
    b = np.matrix(np.array([0.0]).reshape(-1,1))

            
    return Q, p, G, h, A, b

