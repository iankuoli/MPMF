import numpy as np
from scipy.sparse import *
from scipy.special import *


def similarity(matX, matY, type, r=1):
    if type == 'rbf':

        matRet = np.dot(matX * matX.T) + np.dot(matY * matY.T) + -2 * np.dot(matX * matY.T)
        h = np.std(np.reshape(matRet, matRet.shape[0] * matRet.shape[1], 1)) / 20
        matRet = np.exp(- matRet / h)
        return csr_matrix(matRet)

    elif type == 'cos':

        matRet = np.dot(matX, matY.T)
        matD_X = np.sqrt(np.diag(np.dot(matX, matX.T)))
        matD_Y = np.sqrt(np.diag(np.dot(matY, matY.T)))
        matRet = ((matRet / matD_Y).T / matD_X).T
        return csr_matrix(matRet)

    elif type == 'gamma':

        matX += r
        matY += r
        X_denominator = gammaln(matX)
        Y_denominator = gammaln(matY)

        matRet = np.zeros((matX.shape[0], matX.shape[0]))

        for i in range(matX.shape[0]):
            ret = gammaln(0.5 * (matY + matX[i, :])) - 0.5 * X_denominator[i, :] - 0.5 * Y_denominator
            matRet[i, :] = sum(ret.T)

        matRet = np.exp(matRet / matX.shape[1])
        return csr_matrix(matRet)
