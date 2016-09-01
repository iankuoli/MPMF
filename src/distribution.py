import numpy as np
from scipy import *
from scipy.sparse import *
from scipy.special import *


def log_poisson(X, Theta, Beta):
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

    cap_x = log(Theta.dot(Beta))
    
    [x_X, y_X, v_X] = find(X)

    l = l - sum(cap_x)
    
    vecT = np.zeros((x_X.shape[0]))
    for i in range(v_X.shape[0]):

        l += cap_x[x_X[i], y_X[i]] - v_X[i] * log(v_X[i])

    return l
