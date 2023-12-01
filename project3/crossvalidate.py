"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
ktype : (linear, rbf, polynomial)
Cs   : interval of regularization constant that should be tried out
paras: interval of kernel parameters that should be tried out

Output:
bestC: best performing constant C
bestP: best performing kernel parameter
lowest_error: best performing validation error
errors: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)

Trains an SVM classifier for all combination of Cs and paras passed and identifies the best setting.
This can be implemented in many ways and will not be tested by the autograder. You should use this
to choose good parameters for the autograder performance test on test data. 
"""

import numpy as np
import math
from trainsvm import trainsvm

# def crossvalidate(xTr, yTr, ktype, Cs, paras):
    # bestC, bestP, lowest_error = 0, 0, np.inf
    # errors = np.zeros((len(paras),len(Cs)))
    # a,n = xTr.shape

    # for i in range(len(paras)):
    #     p = paras[i]
    #     for j in range(len(Cs)):
    #         c = Cs[j]
    #         svmclassify = trainsvm(xTr, yTr, c, ktype, p)
    #         preds = svmclassify(xTr)
    #         errors[i,j] = np.mean(preds != yTr)

    #         if errors[i,j] < lowest_error:
    #             bestC = c
    #             bestP = p 
    #             lowest_error = errors[i,j]
        # return bestC, bestP, lowest_error, errors

def crossvalidate(xTr, yTr, ktype, Cs, paras):
    bestC, bestP, lowest_error = 0, 0, np.inf
    errors = np.zeros((len(paras),len(Cs)))
    
    # YOUR CODE HERE
    n = yTr.shape[0]
    for c_index, c in enumerate(Cs):
        for p_index, p in enumerate(paras):
            classifier = trainsvm(xTr,yTr, c, ktype, p)
            predicts = classifier(xTr)
            err = (1 - np.sum(predicts == yTr) / n)
            if err < lowest_error:
                lowest_error = err
                bestC = c
                bestP = p
            
    return bestC, bestP, lowest_error, errors

    



    