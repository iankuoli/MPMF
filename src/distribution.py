import numpy as np
from scipy import *
from scipy.sparse import *
from scipy.special import *


def log_Poisson(X, Theta, Beta):
    """
     Calculate the log likelihood with the Poisson distribution (X ~ A * B)
    """
    [m, n] = X.shape
    if Theta.shape[0] != m:
        print("Dimension of Theta is wrong.")
        return

    if Beta.shape[1] != n:
        print("Dimension of Beta is wrong.")
        return

    if Theta.shape[1] != Beta.shape[0]:
        print("Dimension of latent parameter is different.")
        return
    
    l = 0
    
    [x_X, y_X, v_X] = find(X)

    l = l - np.sum(Theta, 0).dot(np.sum(Beta, 1))
    
    vecT = np.zeros((x_X.shape[0]))
    for i in range(v_X.shape[0]):

        vecT[i] = Theta[x_X[i], :].dot(Beta[:, y_X[i]])

    l += np.dot(v_X, log(vecT))

    return l
