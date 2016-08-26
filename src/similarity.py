import numpy as np
from scipy.sparse import *
from scipy.special import *


def similarity(mat_x, mat_y, type, r=1):
    if type == 'rbf':

        mat_ret = np.dot(mat_x * mat_x.T) + np.dot(mat_y * mat_y.T) + -2 * np.dot(mat_x * mat_y.T)
        h = np.std(np.reshape(mat_ret, mat_ret.shape[0] * mat_ret.shape[1], 1)) / 20
        mat_ret = np.exp(- mat_ret / h)
        return csr_matrix(mat_ret)

    elif type == 'cos':

        mat_ret = np.dot(mat_x, mat_y.T)
        matD_X = np.sqrt(np.diag(np.dot(mat_x, mat_x.T)))
        matD_Y = np.sqrt(np.diag(np.dot(mat_y, mat_y.T)))
        mat_ret = ((mat_ret / matD_Y).T / matD_X).T
        return csr_matrix(mat_ret)

    elif type == 'gamma':

        mat_x = mat_x.todense()
        mat_y = mat_y.todense()

        mat_x += r
        mat_y += r
        x_denominator = gammaln(mat_x)
        y_denominator = gammaln(mat_y)

        mat_ret = np.zeros((mat_x.shape[0], mat_x.shape[0]))

        for i in range(mat_x.shape[0]):
            ret = gammaln(0.5 * (mat_y + mat_x[i, :])) - 0.5 * x_denominator[i, :] - 0.5 * y_denominator
            mat_ret[i, :] = sum(ret.T)

        mat_ret = np.exp(mat_ret / mat_x.shape[1])
        return csr_matrix(mat_ret)

    elif type == 'gamma2':

        x_nz_denominator = csr_matrix((gammaln(find(mat_x)[2] + r), (find(mat_x)[0], find(mat_x)[1])), mat_x.shape)
        y_nz_denominator = csr_matrix((gammaln(find(mat_y)[2] + r), (find(mat_y)[0], find(mat_y)[1])), mat_y.shape)

        mat_ret = np.zeros((mat_x.shape[0], mat_x.shape[0]))
        mat_norm = np.zeros((mat_x.shape[0], mat_x.shape[0]))

        for i in range(mat_x.shape[0]):
            nz_mean = 0.5 * (mat_y + np.ones((mat_y.shape[0], 1)) * mat_x[i, :])
            ret = csr_matrix((gammaln(find(nz_mean)[2] + r), (find(nz_mean)[0], find(nz_mean)[1])), nz_mean.shape)
            ret += -0.5 * np.ones((mat_y.shape[0], 1)) * x_nz_denominator[i, :] - 0.5 * y_nz_denominator

            mat_norm[i, :] = np.sum(nz_mean > 0, 1).T
            mat_ret[i, :] = np.squeeze(np.asarray(np.sum(ret, 1)))

        mat_ret = np.exp(mat_ret / mat_norm)
        # mat_ret = np.exp(mat_ret / mat_x.shape[1])
        return csr_matrix(mat_ret)
