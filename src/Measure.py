import numpy as np
from scipy import *
from scipy.sparse import *
from scipy.special import *


def accuracy_for_clustering(vec_label, vec_pred, K):
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


def precision_recall_at_k(vec_label, vec_predict, k):
    label_set = set(np.argpartition(vec_label, -k)[-k:])
    predict_set = set(np.argpartition(vec_predict, -k)[-k:])

    precision = len(set.intersection(label_set, predict_set)) / len(predict_set)
    recall = len(set.intersection(label_set, predict_set)) / vec_label.nnz

    return precision, recall

# def ndcg_at_k(vec_label, vec_predict, k):


# def rmse(vec_label, vec_pred):
