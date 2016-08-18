import numpy as np
from scipy import *
from scipy.sparse import *
from scipy.special import *


def accuracy(vec_label, vec_pred, K):

    N = vec_label.shape[0]

    matLabel = csr_matrix((np.ones((N, 1)), (list(range(N)), vec_label)), shape=(N, K))
    matPred = csr_matrix((np.ones((N, 1)), (list(range(N)), vec_pred)), shape=(N, K))

    accurate_instance = 0

    for k in range(K):

        match = matLabel[:, k].T * matPred
        [max_match_val, max_match_idx] = max(match)

        accurate_instance = accurate_instance + max_match_val
        matPred[:, max_match_idx] = []

    return accurate_instance / N

