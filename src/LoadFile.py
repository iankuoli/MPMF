import numpy as np
from scipy import *
from scipy.sparse import *


def load_small_toy(file_path):
    """
    Load User/Item recording data
    :param file_path: (user_id,item_id,freq)
    :return: matX, i.e., user/item matrix
    """

    list_users = []
    list_items = []
    list_freq = []

    with open(file_path, 'r', encoding='UTF-8') as data:
        for line in data:
            l = line.strip('\n').split(',')
            user = int(l[0]) - 1
            item = int(l[1]) - 1
            freq = float(l[2])
            list_users.append(user)
            list_items.append(item)
            list_freq.append(freq)

    mat_x = csr_matrix((list_freq, (list_users, list_items)))
    return mat_x


def load_iris(file_path):
    """
    Load IRIS data
    :param file_path: (attr1,attr2,attr3,attr4,label)
    :return: matX
    """

    list_data = []
    vec_label = []

    dict_label = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}

    with open(file_path, 'r', encoding='UTF-8') as data:
        for line in data:
            l = line.strip('\n').split(',')
            list_data.append(list(map(float, l[0:4])))
            vec_label.append(dict_label[l[4]])

    mat_x = csr_matrix(list_data)
    return mat_x, vec_label


def load_yeast(file_path):
    """
    Load YEAST data
    :param file_path:
    :return:
    """
    list_data = []
    vec_label = []

    dict_label = {'CYT': 1, 'NUC': 2, 'MIT': 3, 'ME3': 4, 'ME2': 5, 'ME1': 6, 'EXC': 7, 'VAC': 8, 'POX': 9, 'ERL': 10}

    with open(file_path, 'r', encoding='UTF-8') as data:
        for line in data:
            l = line.strip('\n').split(',')
            list_data.append(list(map(float, l[1:9])))
            vec_label.append(dict_label[l[9]])

    mat_x = csr_matrix(list_data)
    return mat_x, vec_label


def load_last_fm(ui_path, user_profile_path, item_profile_path):
    """
    Load User/Item recording data
    :param file_path: (user_id,item_id,freq)
    :return: matX, i.e., user/item matrix
    """

    list_users = []
    list_items = []
    list_freq = []

    list_user_name2id = dict()
    list_user_id2name = dict()
    list_item_name2id = dict()
    list_item_id2name = dict()

    with open(user_profile_path, 'r', encoding='UTF-8') as u_data:
        for line in u_data:
            l = line.strip('\n').split(',')
            user_id = int(l[0])
            user_name = l[1]
            list_user_name2id[user_name] = user_id
            list_user_id2name[user_id] = user_name

    with open(item_profile_path, 'r', encoding='UTF-8') as i_data:
        for line in i_data:
            l = line.strip('\n').split(',')
            item_id = int(l[0])
            item_name = l[1]
            list_item_name2id[item_name] = item_id
            list_item_id2name[item_id] = item_name

    with open(ui_path, 'r', encoding='UTF-8') as data:
        for line in data:
            l = line.strip('\n').split(',')
            user = int(l[0])
            item = int(l[1])
            freq = float(l[2])
            list_users.append(user)
            list_items.append(item)
            list_freq.append(freq)

    mat_x = csr_matrix((list_freq, (list_users, list_items)))
    return mat_x, list_user_name2id, list_user_id2name, list_item_name2id, list_item_id2name


def load_imdb(ui_path):
    matX = csr.csr_matrix((1, 1))
    return matX

def load_UCI_MSD(ui_path):
    with open(ui_path, 'r', encoding='UTF-8') as ui_data:
        for line in ui_data:
            print(line)


def load_EchoNestTaste(ui_path):

    dict_user2id = dict()
    dict_item2id = dict()

    row_index = []
    col_index = []
    val_index = []

    with open(ui_path, 'r', encoding='UTF-8') as ui_data:

        for line in ui_data:
            l = line.strip('\n').split('\t')
            user_key = l[0]
            item_key = l[1]
            count = int(l[2])

            if user_key in dict_user2id.keys():
                uid = dict_user2id[user_key]
            else:
                uid = len(dict_user2id)
                dict_user2id[user_key] = uid

            if item_key in dict_item2id.keys():
                iid = dict_item2id[item_key]
            else:
                iid = len(dict_item2id)
                dict_item2id[item_key] = iid

            row_index.append(uid)
            col_index.append(iid)
            val_index.append(count)

    nnz = len(val_index)
    mask = np.random.randint(100, size=nnz)

    # rand_num < 21: for testing
    # rand_num == 21: for validation / tune parameters
    # rand_num > 21: for training

    idx_test = mask < 21
    idx_valid = mask == 21
    idx_train = mask > 21

    matX_train = csr_matrix((np.compress(idx_train, val_index, axis=0),
                            (np.compress(idx_train, row_index, axis=0), np.compress(idx_train, col_index, axis=0))))

    matX_test = csr_matrix((np.compress(idx_test, val_index, axis=0),
                             (np.compress(idx_test, row_index, axis=0), np.compress(idx_test, col_index, axis=0))))

    matX_valid = csr_matrix((np.compress(idx_valid, val_index, axis=0),
                            (np.compress(idx_valid, row_index, axis=0), np.compress(idx_valid, col_index, axis=0))))

    return matX_train, matX_test, matX_valid
